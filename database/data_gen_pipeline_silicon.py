import cadquery as cq
import random
import json
import time
import os
import gc  # 新增：用于垃圾回收
import math
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 注册硅基流动 (https://cloud.siliconflow.cn/) 获取 Key
API_KEY = "sk-bakrzvjqhtfdozmltgmguwkuiyohkxnnmzwkojujtstekprc"  # 替换你的 Key
API_BASE = "https://api.siliconflow.cn/v1"       # API 地址
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"           # 模型名称

OUTPUT_FILE = "complex_cnc_dataset_qwen2.jsonl"
NUM_SAMPLES = 10000  # 本次运行计划新增生成的数量
# ===========================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

class ComplexShapeGenerator:
    """
    复杂几何生成器：支持多种基体 + 多工序叠加
    修改版：动态调整参数范围以避免几何内核崩溃
    """
    def r_float(self, min_v, max_v):
        # 增加保护，防止 min > max
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        return round(random.uniform(min_v, max_v), 2)

    def generate_base(self):
        base_type = random.choice(["box", "cylinder", "tube", "polygon"])
        code = ""
        skeleton = ""
        params = []
        dims = {"type": base_type}

        if base_type == "box":
            L = self.r_float(30, 100)
            W = self.r_float(30, 100)
            H = self.r_float(10, 40)
            dims.update({"L": L, "W": W, "H": H})
            code = f"result = cq.Workplane('XY').box({L}, {W}, {H})"
            skeleton = f"result = cq.Workplane('XY').box([ARG], [ARG], [ARG])"
            params = [L, W, H]

        elif base_type == "cylinder":
            R = self.r_float(15, 40)
            H = self.r_float(10, 50)
            dims.update({"R": R, "H": H})
            code = f"result = cq.Workplane('XY').cylinder({H}, {R})"
            skeleton = f"result = cq.Workplane('XY').cylinder([ARG], [ARG])"
            params = [H, R]

        elif base_type == "tube":
            R_outer = self.r_float(20, 50)
            thickness = self.r_float(5, 15)
            R_inner = round(R_outer - thickness, 1)
            H = self.r_float(20, 60)
            dims.update({"R": R_outer, "H": H, "R_inner": R_inner, "thickness": thickness})
            code = f"result = cq.Workplane('XY').tube({H}, {R_outer}, {R_inner})"
            skeleton = f"result = cq.Workplane('XY').tube([ARG], [ARG], [ARG])"
            params = [H, R_outer, R_inner]

        elif base_type == "polygon":
            n_sides = random.choice([6, 8])
            diameter = self.r_float(20, 60)
            H = self.r_float(10, 40)
            # 估算多边形的边长，用于后续倒角安全检查
            side_len = diameter * math.sin(math.pi / n_sides)
            dims.update({"R": diameter/2, "H": H, "side_len": side_len})
            code = f"result = cq.Workplane('XY').polygon({n_sides}, {diameter}).extrude({H})"
            skeleton = f"result = cq.Workplane('XY').polygon([ARG], [ARG]).extrude([ARG])"
            params = [float(n_sides), diameter, H]

        return code, skeleton, params, dims

    def add_feature(self, code, skel, params, dims):
        features = ["top_hole", "top_pocket", "top_boss", "fillet", "chamfer"]
        
        # 只有 Box 支持侧面孔（其他形状侧面不好定位）
        if dims["type"] == "box":
            valid_features = features + ["side_hole"]
        else:
            valid_features = features 

        # Tube 不适合做 pocket，容易切穿内壁导致崩溃，排除 top_pocket
        if dims["type"] == "tube" and "top_pocket" in valid_features:
            valid_features.remove("top_pocket")

        choice = random.choice(valid_features)
        
        # === 动态计算安全限制 ===
        # 计算当前几何体的最小特征尺寸，用于限制倒角和圆角
        if dims["type"] == "box":
            min_dim = min(dims["L"], dims["W"], dims["H"])
            # 平面尺寸限制（用于孔和槽）
            plane_limit = min(dims["L"], dims["W"])
        elif dims["type"] == "cylinder":
            min_dim = min(dims["R"], dims["H"])
            plane_limit = dims["R"] * 2
        elif dims["type"] == "tube":
            # Tube 的最小尺寸由壁厚决定
            min_dim = min(dims["thickness"], dims["H"])
            plane_limit = dims["thickness"] * 2 # 限制在壁厚上操作
        elif dims["type"] == "polygon":
            min_dim = min(dims.get("side_len", 10), dims["H"])
            plane_limit = dims["R"] * 1.5

        # =======================

        if choice == "top_hole":
            # 限制孔径在平面的 60% 以内
            max_r = plane_limit * 0.3 
            # 如果是 Tube，孔必须更小以免破壁
            if dims["type"] == "tube":
                max_r = dims["thickness"] * 0.4
            
            r = self.r_float(1, max(1.5, max_r))
            code += f".faces('>Z').workplane().hole({r*2})"
            skel += f".faces('>Z').workplane().hole([ARG])"
            params.append(r*2)

        elif choice == "top_pocket":
            # 仅 Box 和 Polygon 进入此分支 (Tube 已排除)
            limit_w = plane_limit * 0.7
            w = self.r_float(5, max(5.1, limit_w))
            h = self.r_float(5, max(5.1, limit_w))
            # 深度限制：不要切穿到底，留 10% 
            max_d = dims["H"] * 0.9
            d = self.r_float(2, max(2.1, max_d))
            
            code += f".faces('>Z').workplane().rect({w}, {h}).cutBlind(-{d})"
            skel += f".faces('>Z').workplane().rect([ARG], [ARG]).cutBlind(-[ARG])"
            params.extend([w, h, d])

        elif choice == "top_boss":
            # Boss 半径
            limit_r = plane_limit * 0.3
            r = self.r_float(2, max(2.5, limit_r))
            h = self.r_float(2, 10)
            code += f".faces('>Z').workplane().circle({r}).extrude({h})"
            skel += f".faces('>Z').workplane().circle([ARG]).extrude([ARG])"
            params.extend([r, h])

        elif choice == "side_hole" and dims["type"] == "box":
            # 侧面孔限制
            limit_side = min(dims["W"], dims["H"]) # 假设在 >X 面，受 W 和 H 限制(近似)
            r = self.r_float(2, max(2.1, limit_side * 0.3))
            code += f".faces('>X').workplane().hole({r*2})"
            skel += f".faces('>X').workplane().hole([ARG])"
            params.append(r*2)

        elif choice == "fillet":
            # 【关键修改】Fillet 是崩溃之源
            # 半径必须非常小，通常 < 最小边长的 10%
            max_fillet = min_dim * 0.1
            # 绝对上限设为 5.0，防止对大物体生成过大圆角
            max_fillet = min(5.0, max_fillet)
            
            r = self.r_float(0.5, max(0.6, max_fillet))
            
            # Tube 内外倒角容易出错，Box/Polygon 比较安全
            # 使用 try/catch 保护在外部循环中，这里尽量生成安全的数值
            code += f".edges('|Z').fillet({r})"
            skel += f".edges('|Z').fillet([ARG])"
            params.append(r)

        elif choice == "chamfer":
            # 【关键修改】Chamfer 同理
            max_chamfer = min_dim * 0.1
            max_chamfer = min(3.0, max_chamfer)
            
            d = self.r_float(0.5, max(0.6, max_chamfer))
            code += f".faces('>Z').edges().chamfer({d})"
            skel += f".faces('>Z').edges().chamfer([ARG])"
            params.append(d)

        return code, skel, params

    def generate_complex_sample(self):
        code, skel, params, dims = self.generate_base()
        num_features = random.randint(1, 3) # 减少特征数量上限，降低叠加崩溃风险
        for _ in range(num_features):
            temp_code, temp_skel, temp_params = self.add_feature(code, skel, list(params), dims)
            # 预检代码是否可执行，这层保护能捕获 Python 级错误，但捕获不了 C++ Segfault
            # 所以上面的 add_feature 必须足够"安全"
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{temp_code}", {}, local_vars)
                code = temp_code
                skel = temp_skel
                params = temp_params
            except Exception:
                # 如果这个特征添加失败，就跳过它，继续下一个特征或返回当前状态
                pass
        return {"code": code, "skeleton": skel, "params": params}

def generate_text_with_qwen(code_str):
    prompt = (
        "You are a mechanical engineer. "
        "Describe the geometry created by this Python CadQuery code in one sentence. "
        "Requirements:\n"
        "1. Start with the base shape (Box, Cylinder, Tube, or Polygon).\n"
        "2. Mention every modification (hole, pocket, fillet).\n"
        "3. **CRITICAL**: Include ALL numbers found in the code (dimensions, radius, depth).\n"
        "4. No code explanations, just the physical description.\n\n"
        f"Code:\n{code_str}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"SiliconFlow API Error: {e}")
        return None

# ===========================================
# 核心修改区域：获取文件中最后一个ID
# ===========================================
def get_start_id(filename):
    """读取文件最后一行，解析ID，如果文件不存在则返回0"""
    start_id = 0
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:  # 使用二进制模式读取以支持 seek
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                
                last_line = f.readline().decode().strip()
                if last_line:
                    try:
                        data = json.loads(last_line)
                        start_id = data.get("id", -1) + 1
                        print(f"📂 发现已有数据，将从 ID {start_id} 继续追加...")
                    except json.JSONDecodeError:
                        print("⚠️ 最后一行 JSON 解析失败，将从 ID 0 开始")
        except Exception as e:
            print(f"⚠️ 读取文件出错: {e}，将从 ID 0 开始")
    else:
        print("📂 文件不存在，将创建新文件并从 ID 0 开始...")
    return start_id

def main():
    generator = ComplexShapeGenerator()
    
    # 1. 获取起始 ID
    start_id = get_start_id(OUTPUT_FILE)
    
    print(f"🚀 开始生成 CNC 数据集 (安全参数版)")
    print(f"📡 模型: {MODEL_NAME}")
    print(f"➕ 本次计划新增: {NUM_SAMPLES} 条")
    print(f"🔢 ID 范围: {start_id} -> {start_id + NUM_SAMPLES - 1}")
    
    # success_count 代表本次运行成功的数量
    success_count = 0
    
    # 2. 使用 append 模式 'a' 打开文件
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        
        while success_count < NUM_SAMPLES:
            # 2.1 生成几何
            geo_data = generator.generate_complex_sample()
            
            # 2.2 最终完整性检查
            try:
                local_vars = {}
                # 再次执行以确保最终组合没有造成拓扑错误
                exec(f"import cadquery as cq\n{geo_data['code']}", {}, local_vars)
                if 'result' not in local_vars: continue
                # 可选：检查体积是否有效 (有时会返回负体积或崩坏的几何)
                if local_vars['result'].val().Volume() < 1e-6: continue
            except Exception:
                continue

            # 2.3 调用 Qwen 生成描述
            description = generate_text_with_qwen(geo_data['code'])
            
            if description:
                current_id = start_id + success_count
                
                dataset_item = {
                    "id": current_id,
                    "text": description,
                    "code": geo_data['code'],
                    "skeleton": geo_data['skeleton'],
                    "params": geo_data['params']
                }
                
                f.write(json.dumps(dataset_item, ensure_ascii=False) + "\n")
                
                # 强制刷新缓冲区，确保崩溃时数据已写入
                f.flush() 
                
                success_count += 1
                pbar.update(1)
            
            # 2.4 定期垃圾回收，清理泄露的 semaphore
            if success_count % 50 == 0:
                gc.collect()
            
        pbar.close()

    print(f"\n✅ 生成完成！已追加到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
