import cadquery as cq
import random
import json
import os
import gc
import math
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 注册硅基流动 (https://cloud.siliconflow.cn/) 获取 Key
API_KEY = "sk-bakrzvjqhtfdozmltgmguwkuiyohkxnnmzwkojujtstekprc"  # 替换你的 Key
API_BASE = "https://api.siliconflow.cn/v1"       # API 地址
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"           # 模型名称

OUTPUT_FILE = "complex_cnc_dataset_qwen2-complex.jsonl"
NUM_SAMPLES = 10000  # 本次运行计划新增生成的数量
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ---------- 工具：确保单个实体 ----------
def ensure_single_solid(local_vars):
    """
    合并并检查是否仅有 1 个实体；否则抛出异常以便上层跳过。
    """
    wp = local_vars["result"].combineSolids()
    if len(wp.vals()) != 1:
        raise ValueError("multi-body detected")
    return wp

class ComplexShapeGenerator:
    """
    复杂几何生成器：更多基体 + 多工序 + 多面分布 + 安全约束
    """

    def r_float(self, min_v, max_v):
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        return round(random.uniform(min_v, max_v), 2)

    # ---------- 基体生成 ----------
    def generate_base(self):
        base_type = random.choice([
            "box", "cylinder", "tube", "polygon",
            "frustum", "ellipse_prism"
        ])
        params = []
        dims = {"type": base_type}

        if base_type == "box":
            L = self.r_float(30, 120)
            W = self.r_float(30, 120)
            H = self.r_float(12, 45)
            core_expr = f"cq.Workplane('XY').box({L}, {W}, {H})"
            core_skel = "cq.Workplane('XY').box([ARG], [ARG], [ARG])"
            params.extend([L, W, H])
            dims.update({"L": L, "W": W, "H": H, "footprint_L": L, "footprint_W": W})

        elif base_type == "cylinder":
            R = self.r_float(18, 50)
            H = self.r_float(14, 60)
            core_expr = f"cq.Workplane('XY').cylinder({H}, {R})"
            core_skel = "cq.Workplane('XY').cylinder([ARG], [ARG])"
            params.extend([H, R])
            dims.update({"R": R, "H": H, "footprint_L": 2*R, "footprint_W": 2*R})

        elif base_type == "tube":
            R_outer = self.r_float(24, 60)
            thickness = self.r_float(5, 14)
            R_inner = round(R_outer - thickness, 2)
            H = self.r_float(24, 70)
            core_expr = f"cq.Workplane('XY').tube({H}, {R_outer}, {R_inner})"
            core_skel = "cq.Workplane('XY').tube([ARG], [ARG], [ARG])"
            params.extend([H, R_outer, R_inner])
            dims.update({"R": R_outer, "H": H, "thickness": thickness,
                         "footprint_L": 2*R_outer, "footprint_W": 2*R_outer})

        elif base_type == "polygon":
            n_sides = random.choice([6, 8])
            diameter = self.r_float(26, 70)
            H = self.r_float(14, 45)
            side_len = diameter * math.sin(math.pi / n_sides)
            core_expr = f"cq.Workplane('XY').polygon({n_sides}, {diameter}).extrude({H})"
            core_skel = "cq.Workplane('XY').polygon([ARG], [ARG]).extrude([ARG])"
            params.extend([float(n_sides), diameter, H])
            dims.update({"R": diameter/2, "H": H, "side_len": side_len,
                         "footprint_L": diameter, "footprint_W": diameter})

        elif base_type == "frustum":
            R1 = self.r_float(16, 45)
            R2 = self.r_float(10, R1)
            H = self.r_float(18, 55)
            core_expr = (
                f"cq.Workplane('XY').circle({R1}).workplane(offset={H}).circle({R2}).loft(combine=True)"
            )
            core_skel = "cq.Workplane('XY').circle([ARG]).workplane(offset=[ARG]).circle([ARG]).loft(combine=True)"
            params.extend([R1, H, R2])
            footprint = 2 * max(R1, R2)
            dims.update({"R1": R1, "R2": R2, "H": H, "footprint_L": footprint, "footprint_W": footprint})

        elif base_type == "ellipse_prism":
            a = self.r_float(20, 55)
            b = self.r_float(14, a)
            H = self.r_float(14, 45)
            core_expr = f"cq.Workplane('XY').ellipse({a}, {b}).extrude({H})"
            core_skel = "cq.Workplane('XY').ellipse([ARG], [ARG]).extrude([ARG])"
            params.extend([a, b, H])
            dims.update({"a": a, "b": b, "H": H, "footprint_L": 2*a, "footprint_W": 2*b})

        # 可选台座
        add_pad = random.random() < 0.35
        if add_pad:
            pad_margin = self.r_float(4, 14)
            pad_t = self.r_float(3, 10)
            base_L = dims.get("footprint_L", 60)
            base_W = dims.get("footprint_W", base_L)
            pad_expr = (
                f"cq.Workplane('XY').rect({base_L + 2*pad_margin}, {base_W + 2*pad_margin})"
                f".extrude({pad_t}).translate((0, 0, -{pad_t/2}))"
            )
            pad_skel = (
                "cq.Workplane('XY').rect([ARG], [ARG]).extrude([ARG]).translate((0, 0, -[ARG]/2))"
            )
            params.extend([base_L + 2*pad_margin, base_W + 2*pad_margin, pad_t])
            code = f"result = ({core_expr}.union({pad_expr}))"
            skeleton = f"result = ({core_skel}.union({pad_skel}))"
            dims.update({"pad_t": pad_t})
        else:
            code = f"result = {core_expr}"
            skeleton = f"result = {core_skel}"

        return code, skeleton, params, dims

    # ---------- 面选择 & 尺寸 ----------
    def pick_face(self, dims, op):
        base = dims["type"]
        if op == "ring_groove":
            return random.choice([">Z", "<Z"])
        if base == "box":
            return random.choice([">Z", "<Z", ">X", "<X", ">Y", "<Y"])
        if base in ["polygon", "frustum", "ellipse_prism"]:
            return random.choice([">Z", "<Z"])
        if base in ["cylinder", "tube"]:
            if op in ["top_hole", "circular_array_holes"]:
                return random.choice([">Z", "<Z", ">X", "<X", ">Y", "<Y"])
            else:
                return random.choice([">Z", "<Z"])
        return ">Z"

    def face_plane_limit(self, dims, face, default_plane_limit):
        if dims["type"] != "box":
            return default_plane_limit
        if face in [">Z", "<Z"]:
            return min(dims["L"], dims["W"])
        if face in [">X", "<X"]:
            return min(dims["W"], dims["H"])
        if face in [">Y", "<Y"]:
            return min(dims["L"], dims["H"])
        return default_plane_limit

    def signed_depth(self, face, depth):
        return -depth if face in [">Z", ">X", ">Y"] else depth

    # ---------- 特征生成 ----------
    def add_feature(self, code, skel, params, dims):
        base_type = dims["type"]
        features = [
            "top_hole", "top_pocket", "top_boss",
            "slot", "counterbore", "circular_array_holes",
            "fillet", "chamfer", "ring_groove"
        ]
        if base_type == "tube":
            features = [f for f in features if f not in ["top_pocket", "slot", "circular_array_holes"]]
        if base_type == "box":
            features += ["side_hole", "side_slot"]

        choice = random.choice(features)

        # 全局尺寸估计
        if base_type == "box":
            min_dim = min(dims["L"], dims["W"], dims["H"])
            plane_limit = min(dims["L"], dims["W"])
        elif base_type == "cylinder":
            min_dim = min(dims["R"], dims["H"])
            plane_limit = dims["R"] * 2
        elif base_type == "tube":
            min_dim = min(dims["thickness"], dims["H"])
            plane_limit = dims["thickness"] * 2
        elif base_type == "polygon":
            min_dim = min(dims.get("side_len", 10), dims["H"])
            plane_limit = dims["R"] * 1.6
        elif base_type == "frustum":
            min_dim = min(dims["R2"], dims["H"])
            plane_limit = max(dims["R1"], dims["R2"]) * 2
        elif base_type == "ellipse_prism":
            min_dim = min(dims["b"], dims["H"])
            plane_limit = min(dims["a"], dims["b"]) * 2
        else:
            min_dim = plane_limit = 20

        face = self.pick_face(dims, choice)
        plane_limit = self.face_plane_limit(dims, face, plane_limit)

        if choice == "top_hole":
            max_r = plane_limit * 0.28
            if base_type == "tube":
                max_r = dims["thickness"] * 0.45
            r = self.r_float(1.5, max(2.0, max_r))
            code += f".faces('{face}').workplane().hole({r*2})"
            skel += ".faces('[ARG_FACE]').workplane().hole([ARG])"
            params.extend([face, r*2])

        elif choice == "counterbore":
            pilot = self.r_float(2, plane_limit * 0.2)
            cbore_d = pilot + self.r_float(2, plane_limit * 0.2)
            cbore_depth = self.r_float(1, max(1.2, dims.get("H", plane_limit) * 0.25))
            code += f".faces('{face}').workplane().cboreHole({pilot}, {cbore_d}, {cbore_depth})"
            skel += ".faces('[ARG_FACE]').workplane().cboreHole([ARG], [ARG], [ARG])"
            params.extend([face, pilot, cbore_d, cbore_depth])

        elif choice == "circular_array_holes":
            arr_r = plane_limit * 0.3
            hole_d = self.r_float(2, max(2.2, plane_limit * 0.18))
            count = random.randint(3, 6)
            code += f".faces('{face}').workplane().polarArray({arr_r}, 0, 360, {count}).hole({hole_d})"
            skel += ".faces('[ARG_FACE]').workplane().polarArray([ARG], 0, 360, [ARG]).hole([ARG])"
            params.extend([face, arr_r, count, hole_d])

        elif choice == "slot":
            limit_w = plane_limit * 0.65
            length = self.r_float(8, max(10, limit_w))
            width = self.r_float(3, max(4, plane_limit * 0.25))
            depth = self.r_float(2, max(2.5, dims.get("H", plane_limit) * 0.45))
            signed_d = self.signed_depth(face, depth)
            code += f".faces('{face}').workplane().slot({length}, {width}).cutBlind({signed_d})"
            skel += ".faces('[ARG_FACE]').workplane().slot([ARG], [ARG]).cutBlind([ARG])"
            params.extend([face, length, width, signed_d])

        elif choice == "top_pocket":
            limit_w = plane_limit * 0.75
            w = self.r_float(6, max(6.5, limit_w))
            h = self.r_float(6, max(6.5, limit_w))
            max_d = dims.get("H", plane_limit) * 0.85
            d = self.r_float(2.5, max(3.0, max_d))
            signed_d = self.signed_depth(face, d)
            code += f".faces('{face}').workplane().rect({w}, {h}).cutBlind({signed_d})"
            skel += ".faces('[ARG_FACE]').workplane().rect([ARG], [ARG]).cutBlind([ARG])"
            params.extend([face, w, h, signed_d])

        elif choice == "ring_groove":
            outer = plane_limit * 0.45
            inner = max(outer * 0.55, outer - self.r_float(3, 8))
            depth = self.r_float(1.5, max(2.0, dims.get("H", plane_limit) * 0.25))
            signed_d = self.signed_depth(face, depth)
            code += f".faces('{face}').workplane().circle({outer}).circle({inner}).cutBlind({signed_d})"
            skel += ".faces('[ARG_FACE]').workplane().circle([ARG]).circle([ARG]).cutBlind([ARG])"
            params.extend([face, outer, inner, signed_d])

        elif choice == "top_boss":
            limit_r = plane_limit * 0.32
            r = self.r_float(2.5, max(3.0, limit_r))
            h = self.r_float(3, 12)
            code += f".faces('{face}').workplane().circle({r}).extrude({h})"
            skel += ".faces('[ARG_FACE]').workplane().circle([ARG]).extrude([ARG])"
            params.extend([face, r, h])

        elif choice == "side_hole" and base_type == "box":
            face = random.choice([">X", "<X", ">Y", "<Y"])
            limit_side = min(dims["W"], dims["H"]) if face in [">X", "<X"] else min(dims["L"], dims["H"])
            r = self.r_float(2, max(2.2, limit_side * 0.32))
            code += f".faces('{face}').workplane().hole({r*2})"
            skel += ".faces('[ARG_FACE]').workplane().hole([ARG])"
            params.extend([face, r*2])

        elif choice == "side_slot" and base_type == "box":
            face = random.choice([">X", "<X", ">Y", "<Y"])
            if face in [">X", "<X"]:
                limit_side = min(dims["W"], dims["H"])
            else:
                limit_side = min(dims["L"], dims["H"])
            length = self.r_float(10, max(12, limit_side * 0.8))
            width = self.r_float(3, max(3.5, limit_side * 0.35))
            depth = self.r_float(2, max(2.5, limit_side * 0.6))
            signed_d = self.signed_depth(face, depth)
            code += f".faces('{face}').workplane().slot({length}, {width}).cutBlind({signed_d})"
            skel += ".faces('[ARG_FACE]').workplane().slot([ARG], [ARG]).cutBlind([ARG])"
            params.extend([face, length, width, signed_d])

        elif choice == "fillet":
            max_f = min(5.0, min_dim * 0.12)
            r = self.r_float(0.6, max(0.8, max_f))
            code += f".edges('|Z').fillet({r})"
            skel += ".edges('|Z').fillet([ARG])"
            params.append(r)

        elif choice == "chamfer":
            max_c = min(3.5, min_dim * 0.12)
            d = self.r_float(0.6, max(0.8, max_c))
            code += f".faces('>Z').edges().chamfer({d})"
            skel += ".faces('>Z').edges().chamfer([ARG])"
            params.append(d)

        return code, skel, params

    # ---------- 样本生成 ----------
    def generate_complex_sample(self):
        code, skel, params, dims = self.generate_base()
        num_features = random.randint(4, 6)
        for _ in range(num_features):
            tmp_code, tmp_skel, tmp_params = self.add_feature(code, skel, list(params), dims)
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{tmp_code}", {}, local_vars)
                local_vars["result"] = ensure_single_solid(local_vars)
                code, skel, params = tmp_code, tmp_skel, tmp_params
            except Exception:
                pass
        extra_features = random.randint(1, 2)
        for _ in range(extra_features):
            tmp_code, tmp_skel, tmp_params = self.add_feature(code, skel, list(params), dims)
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{tmp_code}", {}, local_vars)
                local_vars["result"] = ensure_single_solid(local_vars)
                code, skel, params = tmp_code, tmp_skel, tmp_params
            except Exception:
                pass
        return {"code": code, "skeleton": skel, "params": params}

# ---------- 文本生成 ----------
def generate_text_with_qwen(code_str):
    prompt = (
        "You are a mechanical engineer. "
        "Describe the geometry created by this Python CadQuery code in one sentence. "
        "Requirements:\n"
        "1. Start with the base shape (Box, Cylinder, Tube, Polygon, Frustum, Ellipse).\n"
        "2. Mention every modification (hole, pocket, boss, slot, groove, fillet, chamfer).\n"
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

# ---------- ID 续写 ----------
def get_start_id(filename):
    start_id = 0
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
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

# ---------- 主流程 ----------
def main():
    generator = ComplexShapeGenerator()
    start_id = get_start_id(OUTPUT_FILE)

    print(f"🚀 开始生成 CNC 数据集 (多工序 & 多面 & 单体安全版)")
    print(f"📡 模型: {MODEL_NAME}")
    print(f"➕ 本次计划新增: {NUM_SAMPLES} 条")
    print(f"🔢 ID 范围: {start_id} -> {start_id + NUM_SAMPLES - 1}")

    success_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        while success_count < NUM_SAMPLES:
            geo_data = generator.generate_complex_sample()
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{geo_data['code']}", {}, local_vars)
                if 'result' not in local_vars:
                    continue
                wp = ensure_single_solid(local_vars)
                if wp.val().Volume() < 1e-6:
                    continue
            except Exception:
                continue

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
                f.flush()
                success_count += 1
                pbar.update(1)

            if success_count % 50 == 0:
                gc.collect()
        pbar.close()

    print(f"\n✅ 生成完成！已追加到: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()