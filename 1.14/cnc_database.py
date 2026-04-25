import cadquery as cq
import random
import json
import os
import gc
import math
import re
import multiprocessing as mp
import queue
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import signal
import time

from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-bakrzvjqhtfdozmltgmguwkuiyohkxnnmzwkojujtstekprc"
API_BASE = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_FILE = "database/complex_cnc_dataset_v2.jsonl"
HISTORY_STATS_FILE = "database/param_statistics.json"
TEMP_DIR = "database/temp_geometry"

NUM_SAMPLES = 1000
WORKER_TIMEOUT = 30
USE_LLM_TEXT = True
LLM_TIMEOUT = 20
LLM_MAX_RETRIES = 2
INTENT_LANG = "en"

ENABLE_STL_EXPORT = True
ENABLE_VOXELIZATION = True  # 建议禁用（太慢）
ENABLE_POINT_CLOUD = True
VOXEL_PITCH = 5.0
POINT_CLOUD_SAMPLES = 512

SIGMA_MIN = 0.1
SIGMA_MAX = 5.0
EXPLICIT_RATIO = 0.35
COLD_START_UNCERTAINTY = 0.05

ENERGY_THRESHOLD = 12.0
NEGATIVE_SAMPLE_RATIO = 0.15

UNIT = "mm"
# ===========================================

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass


# ✅ LLM 客户端初始化函数（仅在主进程调用）
def init_llm_client():
    """初始化 LLM 客户端"""
    if not API_KEY:
        print("⚠️ 未检测到 API_KEY，LLM 功能将被禁用")
        return None
    
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        print(f"✅ LLM 客户端初始化成功：{MODEL_NAME}")
        return client
    except Exception as e:
        print(f"⚠️ LLM 客户端初始化失败：{e}")
        return None


# ================= 数据结构 =================
@dataclass
class OpLogEntry:
    stage: str
    op: str
    face: Optional[str]
    args: Dict[str, float]
    instance_id: int = 0
    direction: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None


@dataclass
class MetricParam:
    name: str
    mu: float
    sigma: float
    dist: str = "gaussian"
    unit: str = UNIT
    is_explicit: bool = False
    source: str = "learned"


@dataclass
class GeometryExport:
    stl_path: Optional[str] = None
    voxel_array: Optional[List] = None
    point_cloud:  Optional[List] = None
    bounding_box: Optional[Dict] = None
    volume: float = 0.0
    surface_area: float = 0.0


@dataclass
class EnergyCheck:
    score:  float
    passed: bool
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    severity: str = "low"


@dataclass
class SampleGeo:
    code:  str
    skeleton: str
    params: List[Any]
    dims: Dict[str, Any]
    op_log: List[OpLogEntry]
    geometry_export: Optional[GeometryExport] = None


# ================= 辅助类 =================
class ParameterStatistics:
    def __init__(self, stats_file: str):
        self.stats_file = stats_file
        self.stats = self. load()

    def load(self) -> Dict: 
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json. load(f)
        return {}

    def save(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def update(self, param_name: str, value: float):
        if param_name not in self.stats:
            self.stats[param_name] = {"values": [], "mean": 0, "std": 0, "count": 0}

        entry = self.stats[param_name]
        entry["values"].append(value)
        if len(entry["values"]) > 1000:
            entry["values"] = entry["values"][-1000:]

        values = np.array(entry["values"])
        entry["mean"] = float(np.mean(values))
        entry["std"] = float(np.std(values))
        entry["count"] = len(values)

    def get_sigma(self, param_name: str, value: float) -> float:
        if param_name not in self.stats or self.stats[param_name]["count"] < 5:
            return max(SIGMA_MIN, min(abs(value) * COLD_START_UNCERTAINTY, SIGMA_MAX))
        std = self.stats[param_name]["std"]
        return max(SIGMA_MIN, min(std, SIGMA_MAX))


class ExplicitConstraintExtractor:
    @staticmethod
    def extract(intent_text: str, op_log: List[OpLogEntry]) -> Dict[str, float]:
        explicit_map = {}
        pattern1 = r'(\w+)\s*=\s*([0-9.]+)\s*mm'
        for match in re.finditer(pattern1, intent_text, re.IGNORECASE):
            key, val = match.group(1).lower(), float(match.group(2))
            explicit_map[key] = val

        pattern2 = r'(hole|slot|boss|pocket|fillet|chamfer)\s+.*?([0-9.]+)\s*mm'
        for match in re.finditer(pattern2, intent_text, re.IGNORECASE):
            feature, val = match.group(1).lower(), float(match.group(2))
            if feature not in explicit_map:
                explicit_map[f"_temp_{feature}"] = []
            explicit_map[f"_temp_{feature}"].append(val)

        final_map = {}
        for op in op_log:
            op_type = op.op. lower()
            temp_key = f"_temp_{op_type}"
            if temp_key in explicit_map and explicit_map[temp_key]: 
                val = explicit_map[temp_key]. pop(0)
                if "d" in op.args:
                    final_map[f"{op.op}#{op. instance_id}. d"] = val
                elif "r" in op.args:
                    final_map[f"{op.op}#{op.instance_id}.r"] = val
                elif "L" in op.args:
                    final_map[f"{op.op}#{op.instance_id}.L"] = val

        return final_map


# ================= LLM 文本生成模块 =================
class LLMTextGenerator:
    """LLM 文本描述生成器"""

    @staticmethod
    def generate_description(code_str: str, intent_text: str, client_instance) -> Optional[str]:
        """
        生成零件的自然语言描述
        
        Args:
            code_str: CadQuery 代码
            intent_text: 结构化意图文本
            client_instance: OpenAI 客户端实例
        
        Returns:
            自然语言描述，失败返回 None
        """
        if not client_instance or not USE_LLM_TEXT: 
            return None

        if INTENT_LANG == "zh":
            prompt = (
                "你是一个机械工程师。根据以下 Python CadQuery 代码，用一句话描述生成的零件几何形状。\n"
                "要求：\n"
                "1. 从基体形状开始描述\n"
                "2. 提及每个修改操作（孔、槽、凸台、倒角、圆角等）\n"
                "3. 包含代码中的所有尺寸数字\n"
                "4. 根据面选择器（>Z/<Z等）使用一致的方位词（顶部/底部/左/右/前/后）\n"
                "5. 使用毫米单位\n"
                "6. 不要解释代码，只描述物理形状\n\n"
                f"代码：\n{code_str}\n\n"
                f"结构化描述（参考）：{intent_text}"
            )
        else:
            prompt = (
                "You are a mechanical engineer.  Describe the geometry created by this Python CadQuery code in one natural sentence.\n"
                "Requirements:\n"
                "1. Start with the base shape\n"
                "2. Mention every modification (hole, slot, boss, pocket, groove, fillet, chamfer)\n"
                "3. Include ALL numbers from the code (dimensions, radius, depth)\n"
                "4. Be consistent with spatial terms (top/bottom/left/right/front/back) based on face selectors\n"
                "5. Use millimeters\n"
                "6. No code explanations, just physical description\n\n"
                f"Code:\n{code_str}\n\n"
                f"Structured description (reference): {intent_text}"
            )

        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = client_instance.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content":  prompt}],
                    temperature=0.3,
                    max_tokens=300,
                    timeout=LLM_TIMEOUT,
                )

                description = response.choices[0].message.content. strip()

                if len(description) < 20:
                    print(f"⚠️ LLM 输出过短（尝试 {attempt+1}/{LLM_MAX_RETRIES}）")
                    continue

                if "code" in description.lower() or "python" in description.lower():
                    print(f"⚠️ LLM 输出包含代码解释（尝试 {attempt+1}/{LLM_MAX_RETRIES}）")
                    continue

                return description

            except Exception as e:
                print(f"⚠️ LLM 请求失败（尝试 {attempt+1}/{LLM_MAX_RETRIES}): {e}")
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(2)
                continue

        return None


# ================= 几何导出模块 =================
class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


class GeometryExporter:
    @staticmethod
    def export(code:  str, sample_id: int) -> Optional[GeometryExport]:
        """导出几何表示（STL + 点云 + 体素）"""
        try:
            local_vars = {}
            exec(f"import cadquery as cq\n{code}", {}, local_vars)
            wp = local_vars. get("result")
            if not wp: 
                return None

            solid = wp.val()

            # ===== 1. STL 导出 =====
            stl_filename = None
            if ENABLE_STL_EXPORT:
                stl_filename = f"{TEMP_DIR}/sample_{sample_id}. stl"

                try:
                    from OCP.StlAPI import StlAPI_Writer
                    from OCP.BRepMesh import BRepMesh_IncrementalMesh

                    mesh = BRepMesh_IncrementalMesh(solid. wrapped, 0.1)
                    mesh.Perform()

                    stl_writer = StlAPI_Writer()
                    stl_writer.Write(solid.wrapped, stl_filename)

                    if not os.path.exists(stl_filename) or os.path.getsize(stl_filename) < 100:
                        raise ValueError("STL file invalid")

                except Exception as e:
                    print(f"⚠️ STL export failed: {e}")
                    stl_filename = None

            # ===== 2. 基本属性 =====
            try:
                volume = solid.Volume()
            except: 
                volume = 0.0

            # ===== 3. 包围盒 =====
            try:
                bbox = solid.BoundingBox()
                bounding_box = {
                    "xmin": bbox.xmin, "xmax": bbox.xmax,
                    "ymin": bbox. ymin, "ymax": bbox.ymax,
                    "zmin": bbox.zmin, "zmax": bbox.zmax,
                }
            except:
                bounding_box = None

            # ===== 4. 体素化（可选）=====
            voxel_array = None
            if ENABLE_VOXELIZATION and stl_filename and os.path.exists(stl_filename):
                voxel_array = GeometryExporter._voxelize_stl(stl_filename)

            # ===== 5. 点云采样 =====
            point_cloud = None
            if ENABLE_POINT_CLOUD and stl_filename and os.path. exists(stl_filename):
                point_cloud = GeometryExporter._sample_point_cloud(stl_filename)

            return GeometryExport(
                stl_path=stl_filename,
                voxel_array=voxel_array,
                point_cloud=point_cloud,
                bounding_box=bounding_box,
                volume=volume,
                surface_area=0.0,
            )

        except Exception as e:
            import traceback
            print(f"⚠️ Geometry export failed: {e}")
            print(traceback.format_exc())
            return None

    @staticmethod
    def _voxelize_stl(stl_path: str) -> Optional[List]: 
        """体素化"""
        try:
            import trimesh

            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)

            mesh = trimesh.load(stl_path, file_type='stl', force='mesh')

            if not mesh.is_volume:
                print(f"⚠️ Mesh is not watertight, attempting repair")
                mesh. fill_holes()
                mesh.fix_normals()

            voxels = mesh.voxelized(pitch=VOXEL_PITCH)

            if hasattr(signal, 'SIGALRM'):
                signal. alarm(0)

            arr = voxels.matrix. astype(int)
            if arr.size > 125000:
                print(f"⚠️ Voxel array too large:  {arr.shape}")
                return None

            return arr. tolist()

        except (ImportError, TimeoutError) as e:
            print(f"⚠️ Voxelization skipped: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Voxelization failed:  {e}")
            return None
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

    @staticmethod
    def _sample_point_cloud(stl_path: str) -> Optional[List]:
        """点云采样"""
        try:
            import trimesh

            mesh = trimesh.load(stl_path, file_type='stl', force='mesh')

            points, face_indices = trimesh.sample.sample_surface(mesh, POINT_CLOUD_SAMPLES)

            if len(points) < POINT_CLOUD_SAMPLES * 0.5:
                print(f"⚠️ Point cloud sampling yielded too few points: {len(points)}")

            return points.tolist()

        except ImportError: 
            print("⚠️ trimesh not installed, point cloud skipped")
            return None
        except Exception as e:
            print(f"⚠️ Point cloud sampling failed: {e}")
            return None


# ================= 增强物理安检 =================
class EnhancedMachinabilityChecker:
    @staticmethod
    def check(dims: Dict, op_log: List[OpLogEntry], geometry:  Optional[GeometryExport]) -> EnergyCheck:
        reasons = []
        checks = {}
        score = 0.0
        severity = "low"

        if dims. get("H") and dims.get("footprint_L"):
            aspect = dims["H"] / max(1e-3, dims["footprint_L"])
            checks["aspect_ratio"] = aspect
            if aspect > 3.0:
                score += (aspect - 3.0) * 8
                reasons.append(f"Critical aspect ratio {aspect:.2f}")
                severity = "critical"
            elif aspect > 2.0:
                score += (aspect - 2.0) * 4
                reasons.append(f"High aspect ratio {aspect:.2f}")
                severity = max(severity, "high")

        if dims["type"] in ["tube", "ellipse_tube"]:
            t = dims. get("thickness", 0)
            checks["thin_wall"] = t
            if t < 2.5:
                score += (2.5 - t) * 6
                reasons.append(f"Critical thin wall {t:.2f}mm")
                severity = "critical"
            elif t < 4.0:
                score += (4.0 - t) * 3
                reasons.append(f"Thin wall {t:.2f}mm")
                severity = max(severity, "medium")

        deep_holes = []
        for op in op_log:
            if op.op in ["hole", "counterbore", "side_hole"]:
                d = op.args.get("d") or op.args.get("pilot") or op.args.get("cbore_d", 1)
                depth = abs(op.args.get("depth_signed", 0) or op.args.get("depth", 0))
                if depth > 0 and d > 0:
                    ratio = depth / d
                    if ratio > 10:
                        score += (ratio - 10) * 4
                        deep_holes.append(f"{op.op}#{op.instance_id} L/D={ratio:.1f}")
                        severity = "critical"
                    elif ratio > 6:
                        score += (ratio - 6) * 2
                        deep_holes.append(f"{op.op}#{op.instance_id} L/D={ratio:.1f}")
                        severity = max(severity, "high")

        if deep_holes:
            checks["deep_holes"] = deep_holes
            reasons.append(f"Deep holes: {', '.join(deep_holes)}")

        small_features = []
        for op in op_log:
            if op. op in ["hole", "slot", "pocket_circle"]:
                size = op.args.get("d") or op.args.get("r") or op.args.get("W", 999)
                if size < 1.5:
                    score += (1.5 - size) * 5
                    small_features.append(f"{op.op}#{op.instance_id} size={size:.2f}mm")
                    severity = max(severity, "high")

        if small_features: 
            checks["small_features"] = small_features
            reasons. append(f"Tiny features: {', '.join(small_features)}")

        bottom_features = [op for op in op_log if op.face == "<Z" and op.stage == "feature"]
        if len(bottom_features) > 2:
            score += len(bottom_features) * 1.5
            reasons.append(f"Bottom face has {len(bottom_features)} features (clamping risk)")
            severity = max(severity, "medium")
            checks["bottom_interference"] = len(bottom_features)

        side_features = [op for op in op_log if op.face in [">X", "<X", ">Y", "<Y"]]
        if len(side_features) > 3:
            score += len(side_features) * 2
            reasons.append(f"{len(side_features)} side features may need 5-axis")
            severity = max(severity, "medium")
            checks["side_features"] = len(side_features)

        if geometry and geometry.volume: 
            if geometry.volume < 500:
                score += 3
                reasons.append(f"Very small volume {geometry.volume:.1f} mm³")
                severity = max(severity, "medium")
            checks["volume"] = geometry.volume

        passed = score < ENERGY_THRESHOLD

        return EnergyCheck(
            score=round(score, 3),
            passed=passed,
            reasons=reasons,
            checks=checks,
            severity=severity
        )


# ================= 工具函数 =================
def ensure_single_solid(local_vars):
    wp = local_vars["result"]
    solids = wp.vals()
    if not solids:
        raise ValueError("no solids")
    fused = solids[0]
    for s in solids[1:]: 
        fused = fused.fuse(s)
    wp_single = cq.Workplane(obj=fused)
    if len(wp_single.vals()) != 1:
        raise ValueError("multi-body detected")
    return wp_single


# ================= 复杂形状生成器（完整版）=================
class ComplexShapeGenerator:
    """完整版几何生成器"""
    
    def r_float(self, min_v, max_v):
        if min_v > max_v: 
            min_v, max_v = max_v, min_v
        return round(random.uniform(min_v, max_v), 2)

    def generate_base(self):
        """生成基体"""
        base_type = random.choice([
            "box", "rounded_box", "cylinder", "chamfered_cyl",
            "tube", "polygon", "frustum"
        ])
        params = []
        dims = {"type": base_type, "units":  UNIT}
        op_log:  List[OpLogEntry] = []

        if base_type in ["box", "rounded_box"]:
            L = self.r_float(30, 130)
            W = self.r_float(30, 130)
            H = self.r_float(12, 55)
            core_expr = f"cq. Workplane('XY').box({L}, {W}, {H})"
            core_skel = "cq.Workplane('XY').box(L={L}, W={W}, H={H})"
            params.extend([("L", L), ("W", W), ("H", H)])
            dims. update({"L": L, "W": W, "H": H, "footprint_L": L, "footprint_W": W})
            op_log.append(OpLogEntry("base", base_type, None, {"L": L, "W": W, "H": H}, position=(0, 0, 0)))

            if base_type == "rounded_box":
                fil_r = min(6.0, min(L, W, H) * 0.12)
                core_expr += f". edges('|Z').fillet({fil_r})"
                core_skel += ". edges('|Z').fillet(r={fil_r})"
                params.append(("fillet_base_r", fil_r))
                op_log.append(OpLogEntry("base", "fillet_base", None, {"r": fil_r}))

        elif base_type in ["cylinder", "chamfered_cyl"]:
            R = self.r_float(18, 55)
            H = self.r_float(16, 70)
            core_expr = f"cq.Workplane('XY').cylinder({H}, {R})"
            core_skel = "cq. Workplane('XY').cylinder(H={H}, R={R})"
            params.extend([("H", H), ("R", R)])
            dims.update({"R": R, "D": 2 * R, "H":  H, "footprint_L": 2 * R, "footprint_W": 2 * R})
            op_log.append(OpLogEntry("base", base_type, None, {"H": H, "R": R}, position=(0, 0, 0)))

            if base_type == "chamfered_cyl":
                ch = min(4.0, min(R, H) * 0.12)
                core_expr += f".faces('>Z').edges().chamfer({ch})"
                core_skel += ".faces('>Z').edges().chamfer(d={ch})"
                params. append(("chamfer_base_d", ch))
                op_log.append(OpLogEntry("base", "chamfer_base", ">Z", {"d": ch}))

        elif base_type == "tube":
            R_outer = self.r_float(26, 65)
            thickness = self.r_float(5, 14)
            R_inner = round(R_outer - thickness, 2)
            H = self. r_float(24, 80)
            core_expr = f"cq.Workplane('XY').circle({R_outer}).extrude({H}).faces('>Z').workplane().circle({R_inner}).cutThruAll()"
            core_skel = "cq.Workplane('XY').circle(R_outer={R_outer}).extrude(H={H}).faces('>Z').workplane().circle(R_inner={R_inner}).cutThruAll()"
            params.extend([("H", H), ("R_outer", R_outer), ("R_inner", R_inner)])
            dims.update({"R":  R_outer, "D": 2 * R_outer, "H": H, "thickness": thickness, "footprint_L": 2 * R_outer, "footprint_W": 2 * R_outer})
            op_log.append(OpLogEntry("base", base_type, None, {"H": H, "R_outer":  R_outer, "R_inner": R_inner}))

        elif base_type == "polygon":
            n_sides = random.choice([6, 8])
            diameter = self.r_float(28, 80)
            H = self.r_float(16, 55)
            core_expr = f"cq.Workplane('XY').polygon({n_sides}, {diameter}).extrude({H})"
            core_skel = "cq.Workplane('XY').polygon(n={n_sides}, D={diameter}).extrude(H={H})"
            params.extend([("n", float(n_sides)), ("D", diameter), ("H", H)])
            dims.update({"R": diameter / 2, "D": diameter, "H": H, "footprint_L": diameter, "footprint_W": diameter})
            op_log.append(OpLogEntry("base", base_type, None, {"n": n_sides, "D": diameter, "H": H}))

        elif base_type == "frustum":
            R1 = self.r_float(18, 50)
            R2 = self.r_float(10, R1)
            H = self. r_float(20, 60)
            core_expr = f"cq. Workplane('XY').circle({R1}).workplane(offset={H}).circle({R2}).loft(combine=True)"
            core_skel = "cq.Workplane('XY').circle(R1={R1}).workplane(offset={H}).circle(R2={R2}).loft(combine=True)"
            params.extend([("R1", R1), ("H", H), ("R2", R2)])
            footprint = 2 * max(R1, R2)
            dims.update({"R1": R1, "R2":  R2, "H": H, "footprint_L": footprint, "footprint_W": footprint})
            op_log. append(OpLogEntry("base", base_type, None, {"R1": R1, "R2": R2, "H": H}))

        code = f"result = {core_expr}"
        skeleton = f"result = {core_skel}"
        return code, skeleton, params, dims, op_log

    def pick_face(self, dims, op):
        base = dims["type"]
        if base in ["box", "rounded_box"]:
            return random.choice([">Z", "<Z", ">X", "<X", ">Y", "<Y"])
        return random.choice([">Z", "<Z"])

    def signed_depth(self, face, depth):
        return -depth if face in [">Z", ">X", ">Y"] else depth

    def add_feature(self, code, skel, params, dims, op_log):
        """添加特征"""
        base_type = dims["type"]
        features = ["hole", "slot", "top_pocket", "pocket_circle", "top_boss", "fillet", "chamfer"]

        if base_type in ["tube"]:
            features = ["hole", "fillet", "chamfer"]

        choice = random.choice(features)

        if base_type in ["box", "rounded_box"]:
            min_dim = min(dims["L"], dims["W"], dims["H"])
            plane_limit = min(dims["L"], dims["W"])
        elif base_type in ["cylinder", "chamfered_cyl"]:
            min_dim = min(dims["R"], dims["H"])
            plane_limit = dims["R"] * 2
        elif base_type == "tube":
            min_dim = min(dims["thickness"], dims["H"])
            plane_limit = dims["thickness"] * 2
        else:
            min_dim = plane_limit = 20

        face = self.pick_face(dims, choice)

        def log(op_name, args, instance_id=0, direction=None):
            op_log.append(OpLogEntry("feature", op_name, face, args, instance_id=instance_id, direction=direction, position=(0, 0, dims. get("H", 20) / 2)))

        def inst_id(op_name):
            return len([x for x in op_log if x.op == op_name])

        if choice == "hole":
            max_r = plane_limit * 0.28
            r = self.r_float(1.5, max(2.0, max_r))
            iid = inst_id("hole")
            code += f". faces('{face}').workplane().hole({r * 2})"
            skel += f".faces('{face}').workplane().hole(d={{hole#{iid}. d}})"
            params.append((f"hole#{iid}.d", r * 2))
            log("hole", {"d": r * 2}, instance_id=iid, direction=face)

        elif choice == "slot": 
            limit_w = plane_limit * 0.65
            length = self.r_float(10, max(12, limit_w))
            width = self.r_float(3, max(4, plane_limit * 0.25))
            depth = self.r_float(2, max(2.5, min_dim * 0.5))
            signed_d = self.signed_depth(face, depth)
            iid = inst_id("slot")
            code += f".faces('{face}').workplane().slot2D({length}, {width}).cutBlind({signed_d})"
            skel += f".faces('{face}').workplane().slot2D(L={{slot#{iid}.L}}, W={{slot#{iid}.W}}).cutBlind(depth={{slot#{iid}.depth}})"
            params.extend([(f"slot#{iid}.L", length), (f"slot#{iid}.W", width), (f"slot#{iid}.depth", signed_d)])
            log("slot", {"L": length, "W": width, "depth_signed": signed_d}, instance_id=iid, direction=face)

        elif choice == "top_pocket":
            limit_w = plane_limit * 0.75
            w = self.r_float(6, max(7, limit_w))
            h = self.r_float(6, max(7, limit_w))
            d = self.r_float(2.5, max(3.0, min_dim * 0.9))
            signed_d = self.signed_depth(face, d)
            iid = inst_id("top_pocket")
            code += f".faces('{face}').workplane().rect({w}, {h}).cutBlind({signed_d})"
            skel += f".faces('{face}').workplane().rect(w={{top_pocket#{iid}.W}}, h={{top_pocket#{iid}.H}}).cutBlind(depth={{top_pocket#{iid}. depth}})"
            params.extend([(f"top_pocket#{iid}.W", w), (f"top_pocket#{iid}.H", h), (f"top_pocket#{iid}.depth", signed_d)])
            log("top_pocket", {"W":  w, "H": h, "depth_signed": signed_d}, instance_id=iid, direction=face)

        elif choice == "pocket_circle":
            r = self.r_float(3, max(4, plane_limit * 0.25))
            d = self.r_float(2.5, max(3.0, min_dim * 0.85))
            signed_d = self.signed_depth(face, d)
            iid = inst_id("pocket_circle")
            code += f". faces('{face}').workplane().circle({r}).cutBlind({signed_d})"
            skel += f".faces('{face}').workplane().circle(r={{pocket_circle#{iid}.r}}).cutBlind(depth={{pocket_circle#{iid}.depth}})"
            params. extend([(f"pocket_circle#{iid}.r", r), (f"pocket_circle#{iid}.depth", signed_d)])
            log("pocket_circle", {"r": r, "depth_signed": signed_d}, instance_id=iid, direction=face)

        elif choice == "top_boss":
            limit_r = plane_limit * 0.32
            r = self.r_float(2.5, max(3.0, limit_r))
            h = self.r_float(3, 14)
            iid = inst_id("top_boss")
            code += f". faces('{face}').workplane().circle({r}).extrude({h})"
            skel += f". faces('{face}').workplane().circle(r={{top_boss#{iid}.r}}).extrude(h={{top_boss#{iid}.h}})"
            params.extend([(f"top_boss#{iid}.r", r), (f"top_boss#{iid}.h", h)])
            log("top_boss", {"r": r, "h": h}, instance_id=iid, direction=face)

        elif choice == "fillet": 
            max_f = min(5.0, min_dim * 0.12)
            r = self.r_float(0.6, max(0.8, max_f))
            iid = inst_id("fillet")
            code += f".edges('|Z').fillet({r})"
            skel += f". edges('|Z').fillet(r={{fillet#{iid}. r}})"
            params.append((f"fillet#{iid}. r", r))
            log("fillet", {"r": r}, instance_id=iid, direction=None)

        elif choice == "chamfer":
            max_c = min(3.5, min_dim * 0.12)
            d = self.r_float(0.6, max(0.8, max_c))
            iid = inst_id("chamfer")
            code += f".faces('>Z').edges().chamfer({d})"
            skel += f".faces('>Z').edges().chamfer(d={{chamfer#{iid}.d}})"
            params.append((f"chamfer#{iid}.d", d))
            log("chamfer", {"d": d}, instance_id=iid, direction=">Z")

        return code, skel, params, op_log

    def generate_complex_sample(self) -> SampleGeo:
        """生成复杂样本"""
        code, skel, params, dims, op_log = self.generate_base()
        num_features = random.randint(4, 8)

        for _ in range(num_features):
            tmp_code, tmp_skel, tmp_params, tmp_log = self.add_feature(code, skel, list(params), dims, list(op_log))
            try:
                local_vars = {}
                exec(f"import cadquery as cq\n{tmp_code}", {}, local_vars)
                ensure_single_solid(local_vars)
                code, skel, params, op_log = tmp_code, tmp_skel, tmp_params, tmp_log
            except: 
                pass

        return SampleGeo(code=code, skeleton=skel, params=params, dims=dims, op_log=op_log)


# ================= 意图文本生成 =================
FACE_EN = {">Z": "top", "<Z": "bottom", ">X": "right", "<X": "left", ">Y": "front", "<Y": "back", None: ""}
OP_EN = {
    "box": "box", "rounded_box": "filleted box", "cylinder": "cylinder",
    "hole": "hole", "slot": "slot", "pocket_circle": "circular pocket",
    "top_pocket": "rectangular pocket", "top_boss": "boss",
    "fillet":  "fillet", "chamfer":  "chamfer",
}


def build_intent_text_en(op_log: List[OpLogEntry], explicit_params: Dict[str, float]) -> str:
    """构建英文意图文本"""
    phrases = []
    base = next((op for op in op_log if op.stage == "base"), None)
    if base:
        base_name = OP_EN.get(base. op, base.op)
        phrases.append(f"Create a {base_name}.")

    for op in op_log: 
        if op.stage != "feature":
            continue
        face_str = FACE_EN.get(op.face, "")
        where = f" on the {face_str} face" if face_str else ""
        op_name = OP_EN. get(op.op, op. op. replace("_", " "))

        explicit_strs = []
        for k, v in op.args.items():
            param_key = f"{op.op}#{op.instance_id}. {k}"
            if param_key in explicit_params:
                explicit_strs.append(f"{k}={v} mm")

        explicit_part = (" with " + ", ".join(explicit_strs)) if explicit_strs else ""
        phrases.append(f"Add {op_name}{where}{explicit_part}.")

    return " ".join(phrases) if phrases else "Create a complex part."


# ================= 子进程工作函数 =================
def worker_generate_sample(out_queue, sample_id:  int):
    """子进程：生成几何样本（不调用 LLM）"""
    try:
        gen = ComplexShapeGenerator()
        geo = gen.generate_complex_sample()

        local_vars = {}
        exec(f"import cadquery as cq\n{geo.code}", {}, local_vars)
        wp = ensure_single_solid(local_vars)
        if wp. val().Volume() < 1e-6:
            out_queue.put(None)
            return

        geometry_export = GeometryExporter. export(geo. code, sample_id)
        geo. geometry_export = geometry_export

        out_queue.put(geo)
    except Exception: 
        out_queue.put(None)


def get_start_id(filename):
    """获取续写起始 ID"""
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
                    data = json.loads(last_line)
                    start_id = data.get("id", -1) + 1
                    print(f"📂 续写模式：从 ID {start_id} 开始")
        except Exception as e:
            print(f"⚠️ 读取失败:  {e}")
    return start_id


# ✅ 主流程（必须放在 if __name__ == "__main__" 内）
def main():
    """主流程"""
    # ✅ 在主进程中初始化 LLM 客户端
    llm_client = init_llm_client()
    
    print("=" * 60)
    print("🚀 CNC Dataset Generator V2 - 完整版")
    print(f"   LLM 文本生成:  {'✅ ' + MODEL_NAME if USE_LLM_TEXT and llm_client else '❌ 已禁用'}")
    print(f"   STL导出: {'✅' if ENABLE_STL_EXPORT else '❌'}")
    print(f"   体素化: {'✅' if ENABLE_VOXELIZATION else '❌'}")
    print(f"   点云采样: {'✅' if ENABLE_POINT_CLOUD else '❌'}")
    print("=" * 60)

    param_stats = ParameterStatistics(HISTORY_STATS_FILE)
    start_id = get_start_id(OUTPUT_FILE)

    success_count = 0
    negative_count = 0
    llm_success_count = 0
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        pbar = tqdm(total=NUM_SAMPLES, desc="生成样本")

        while success_count < NUM_SAMPLES:
            current_id = start_id + success_count

            q = mp.Queue()
            p = mp.Process(target=worker_generate_sample, args=(q, current_id))
            p.start()
            p.join(WORKER_TIMEOUT)

            if p.is_alive():
                p.terminate()
                p.join()
                q.close()
                continue

            geo:  Optional[SampleGeo] = None
            try:
                geo = q.get(timeout=0.5)
            except queue.Empty:
                pass
            finally:
                q.close()

            if not geo:
                continue

            energy = EnhancedMachinabilityChecker. check(geo. dims, geo.op_log, geo.geometry_export)

            is_negative = not energy.passed
            if is_negative: 
                if random.random() > NEGATIVE_SAMPLE_RATIO:
                    continue
                negative_count += 1

            metric_params = []
            for op in geo.op_log:
                for k, v in op.args. items():
                    param_name = f"{op.op}#{op.instance_id}.{k}"
                    sigma = param_stats.get_sigma(param_name, v)
                    metric_params.append(MetricParam(
                        name=param_name, mu=v, sigma=sigma,
                        source="learned" if param_stats.stats. get(param_name) else "cold_start"
                    ))
                    param_stats.update(param_name, v)

            intent_text_draft = build_intent_text_en(geo.op_log, {})
            explicit_map = ExplicitConstraintExtractor.extract(intent_text_draft, geo.op_log)

            for p in metric_params:
                if p.name in explicit_map:
                    p.sigma = 0.0
                    p. is_explicit = True
                    p.source = "explicit"

            intent_text = build_intent_text_en(geo.op_log, explicit_map)

            # ✅ 在主进程中调用 LLM
            llm_description = None
            if USE_LLM_TEXT and llm_client: 
                llm_description = LLMTextGenerator.generate_description(geo.code, intent_text, llm_client)
                if llm_description:
                    llm_success_count += 1

            dataset_item = {
                "id": current_id,
                "intent_text": intent_text,
                "llm_description": llm_description,
                "symbolic_trace": [asdict(op) for op in geo.op_log],
                "metric_params": [asdict(p) for p in metric_params],
                "explicit_constraints": list(explicit_map.keys()),
                "implicit_params": [p.name for p in metric_params if not p.is_explicit],
                "energy_check": asdict(energy),
                "is_negative_sample": is_negative,
                "code": geo.code,
                "skeleton": geo.skeleton,
                "geometry": asdict(geo.geometry_export) if geo.geometry_export else None,
                "units":  UNIT,
            }

            f.write(json.dumps(dataset_item, ensure_ascii=False) + "\n")
            f.flush()

            success_count += 1
            pbar. update(1)
            pbar.set_postfix({
                "负样本": negative_count,
                "能量":  f"{energy.score:.1f}",
                "特征数": len([x for x in geo.op_log if x.stage == "feature"]),
                "LLM成功率": f"{llm_success_count}/{success_count}"
            })

            if success_count % 50 == 0:
                param_stats.save()
                gc.collect()

        pbar.close()

    param_stats.save()
    print(f"\n✅ 完成！生成 {success_count} 条样本")
    print(f"   负样本: {negative_count} 条")
    print(f"   LLM 成功率: {llm_success_count}/{success_count} ({llm_success_count / success_count * 100:.1f}%)")
    print(f"📊 参数统计:  {HISTORY_STATS_FILE}")
    print(f"💾 数据集: {OUTPUT_FILE}")


# ✅ 保护主流程（防止子进程重复执行）
if __name__ == "__main__": 
    main()