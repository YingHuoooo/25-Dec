import cadquery as cq
import random
import json
import time
import google.generativeai as genai
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 在 https://aistudio.google.com/ 获取免费 Key
GOOGLE_API_KEY = "AIzaSyAY9ckElMsOfWbI2qC05XAUMMdooVsE2JI" 

OUTPUT_FILE = "complex_cnc_dataset.jsonl"
NUM_SAMPLES = 100  # 建议先跑 10 个测试，没问题再跑几千个
# ===========================================

# 配置 Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class ComplexShapeGenerator:
    """
    复杂几何生成器：支持多工序叠加、基体变换
    """
    def __init__(self):
        pass

    def r_float(self, min_v, max_v):
        """生成保留1位小数的随机浮点数"""
        return round(random.uniform(min_v, max_v), 1)

    def generate_base(self):
        """生成基体 (Base Feature)"""
        shape_type = random.choice(["box", "cylinder"])
        params = []
        
        if shape_type == "box":
            L = self.r_float(30, 100)
            W = self.r_float(30, 100)
            H = self.r_float(10, 40)
            # 记录当前物体的包围盒尺寸，用于后续特征的约束
            dims = {"L": L, "W": W, "H": H, "type": "box"}
            
            code = f"result = cq.Workplane('XY').box({L}, {W}, {H})"
            skeleton = f"result = cq.Workplane('XY').box([ARG], [ARG], [ARG])"
            params = [L, W, H]
            
        else: # cylinder
            R = self.r_float(15, 40)
            H = self.r_float(10, 50)
            dims = {"R": R, "H": H, "type": "cylinder"}
            
            code = f"result = cq.Workplane('XY').cylinder({H}, {R})"
            skeleton = f"result = cq.Workplane('XY').cylinder([ARG], [ARG])"
            params = [H, R] # CadQuery cylinder 参数顺序是 height, radius
            
        return code, skeleton, params, dims

    def add_feature(self, code, skel, params, dims):
        """随机添加一个特征工艺"""
        # 定义特征池
        features = ["top_hole", "top_pocket", "top_boss", "fillet", "chamfer"]
        # 如果是圆柱体，某些特征不太好加（为了简化逻辑，圆柱体少加侧面特征）
        if dims["type"] == "cylinder":
            valid_features = ["top_hole", "top_boss", "fillet", "chamfer"]
        else:
            valid_features = features + ["side_hole"]

        choice = random.choice(valid_features)
        
        # --- 1. 顶面钻孔 (Drill Hole) ---
        if choice == "top_hole":
            limit = dims.get("R", min(dims.get("L", 100), dims.get("W", 100))/2)
            r = self.r_float(2, limit * 0.4) # 孔径不要太大
            
            code += f".faces('>Z').workplane().hole({r*2})" # CadQuery hole用的是直径
            skel += f".faces('>Z').workplane().hole([ARG])"
            params.append(r*2)

        # --- 2. 顶面挖矩形槽 (Rectangular Pocket) ---
        elif choice == "top_pocket" and dims["type"] == "box":
            w = self.r_float(5, dims["L"] * 0.6)
            h = self.r_float(5, dims["W"] * 0.6)
            d = self.r_float(2, dims["H"] * 0.5)
            
            code += f".faces('>Z').workplane().rect({w}, {h}).cutBlind(-{d})"
            skel += f".faces('>Z').workplane().rect([ARG], [ARG]).cutBlind(-[ARG])"
            params.extend([w, h, d])

        # --- 3. 顶面凸台 (Circular Boss) ---
        elif choice == "top_boss":
            limit = dims.get("R", min(dims.get("L", 100), dims.get("W", 100))/2)
            r = self.r_float(5, limit * 0.6)
            h = self.r_float(2, 10)
            
            code += f".faces('>Z').workplane().circle({r}).extrude({h})"
            skel += f".faces('>Z').workplane().circle([ARG]).extrude([ARG])"
            params.extend([r, h])

        # --- 4. 侧面钻孔 (Side Hole - 仅针对方块) ---
        elif choice == "side_hole" and dims["type"] == "box":
            # 在 >X 面打孔
            r = self.r_float(2, dims["H"] * 0.3)
            code += f".faces('>X').workplane().hole({r*2})"
            skel += f".faces('>X').workplane().hole([ARG])"
            params.append(r*2)

        # --- 5. 倒圆角 (Fillet - 竖直边) ---
        elif choice == "fillet":
            r = self.r_float(1, 3)
            code += f".edges('|Z').fillet({r})"
            skel += f".edges('|Z').fillet([ARG])"
            params.append(r)

        # --- 6. 倒角 (Chamfer - 顶面边缘) ---
        elif choice == "chamfer":
            d = self.r_float(1, 2)
            code += f".faces('>Z').edges().chamfer({d})"
            skel += f".faces('>Z').edges().chamfer([ARG])"
            params.append(d)

        return code, skel, params

    def generate_complex_sample(self):
        """生成一个包含多个工艺步骤的复杂样本"""
        # 1. 生成基体
        code, skel, params, dims = self.generate_base()
        
        # 2. 随机决定叠加多少个特征 (1 到 4 个)
        num_features = random.randint(1, 4)
        
        # 3. 循环添加特征
        for _ in range(num_features):
            # 这里的 try-except 是为了防止生成的参数导致几何构建失败（比如倒角太大吃掉了整个边）
            # 我们只保留成功的步骤
            temp_code, temp_skel, temp_params = self.add_feature(code, skel, list(params), dims)
            
            try:
                # 尝试编译一下，看看会不会报错
                local_vars = {}
                exec(f"import cadquery as cq\n{temp_code}", {}, local_vars)
                # 如果没报错，确认这个步骤有效，更新主代码
                code = temp_code
                skel = temp_skel
                params = temp_params
            except Exception:
                # 如果报错了（比如几何冲突），就忽略这次尝试，继续下一次循环
                pass
                
        return {
            "code": code,
            "skeleton": skel,
            "params": params
        }

def generate_text_with_gemini(code_str):
    """
    使用 Google Gemini Flash 生成描述
    """
    prompt = (
        "Role: Mechanical Engineer.\n"
        "Task: Describe the 3D geometry created by the provided Python CadQuery code.\n"
        "Requirements:\n"
        "1. Mention the base shape (e.g., box, cylinder) and its dimensions.\n"
        "2. Describe each feature (holes, fillets, chamfers, pockets) applied to it.\n"
        "3. You MUST explicitly state every single number found in the code (dimensions, radii, depths).\n"
        "4. Be concise and imperative (e.g., 'Drill a 5mm hole...').\n"
        "5. Output ONLY the description text.\n\n"
        f"Code Snippet:\n{code_str}"
    )
    
    try:
        # Gemini 免费版有 RPM 限制 (15次/分钟)，所以必须 sleep
        time.sleep(4.5) 
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"\nGemini API Error: {e}")
        # 如果被限流，多睡一会
        time.sleep(10)
        return None

def main():
    generator = ComplexShapeGenerator()
    
    print(f"🚀 开始生成复杂 CNC 数据集，目标: {NUM_SAMPLES} 条")
    print(f"📡 使用模型: Gemini-1.5-Flash (免费版)")
    
    success_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # 使用 tqdm 显示进度条
        pbar = tqdm(total=NUM_SAMPLES)
        
        while success_count < NUM_SAMPLES:
            # 1. 生成几何
            geo_data = generator.generate_complex_sample()
            
            # 2. 再次最终验证 (确保最终结果有 result 对象)
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{geo_data['code']}", {}, local_vars)
                if 'result' not in local_vars:
                    continue
            except Exception:
                continue

            # 3. 调用 Gemini 生成文本
            description = generate_text_with_gemini(geo_data['code'])
            
            if description:
                dataset_item = {
                    "id": success_count,
                    "text": description,
                    "code": geo_data['code'],
                    "skeleton": geo_data['skeleton'],
                    "params": geo_data['params']
                }
                
                f.write(json.dumps(dataset_item, ensure_ascii=False) + "\n")
                success_count += 1
                pbar.update(1)
            else:
                # API 失败时不增加计数
                pass
                
        pbar.close()

    print(f"\n✅ 生成完成！数据已保存至 {OUTPUT_FILE}")
    print("示例样本:")
    print(json.dumps(dataset_item, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()