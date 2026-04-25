import cadquery as cq
import trimesh
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor

def check_machinability(step_file_path, threshold=0.95, ray_samples=5000):
    """
    判断一个 STEP 文件是否适合 3轴 CNC 加工。
    
    原理：
    1. 将 STEP 转为 Mesh。
    2. 在 Mesh 表面均匀采样点。
    3. 向 6 个轴向发射光线 (Ray Casting)。
    4. 如果某个方向上，未被遮挡的点占比超过 threshold，则认为可加工。
    
    Args:
        step_file_path: STEP 文件路径
        threshold: 可见表面积阈值 (0.95 表示允许 5% 的微小遮挡或误差)
        ray_samples: 采样点数量，越高越准但越慢
    
    Returns:
        (bool, str): (是否通过筛选, 最佳加工方向)
    """
    try:
        # 1. 使用 CadQuery 加载 STEP 并转为 Mesh (在内存中完成，不写文件)
        # CadQuery 的 BRep 接口非常稳健
        model = cq.importers.importStep(step_file_path)
        
        # 导出为 STL 格式的字符串流，再由 Trimesh 读取
        # 这种方式避免了复杂的中间文件 IO
        stl_data = run_export_stl(model) 
        mesh = trimesh.load(trimesh.util.wrap_as_stream(stl_data), file_type='stl')
        
        # 2. 预处理：修复简单的网格问题，确保法线正确
        if not mesh.is_watertight:
            # 尝试简单的修复，如果还是破面严重，可能本身就是坏数据，建议丢弃
            mesh.fill_holes()
            
        # 3. 表面采样
        # 采样点 (points) 和 对应的法线 (normals)
        points, face_indices = trimesh.sample.sample_surface(mesh, ray_samples)
        normals = mesh.face_normals[face_indices]
        
        # 为了防止“自遮挡”误判（光线打到发射点自己），将发射点沿法线向外推一点点
        ray_origins = points + normals * 1e-3

        # 4. 定义 6 个主要加工方向 (CNC 通常是 3 轴，工件可以装夹在不同方向)
        directions = [
            [0, 0, 1], [0, 0, -1],
            [0, 1, 0], [0, -1, 0],
            [1, 0, 0], [-1, 0, 0]
        ]
        dir_names = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]

        # 5. 并行光线追踪检测
        # 使用 trimesh 的 ray.intersects_any (底层调用 embree，非常快)
        
        for direction, name in zip(directions, dir_names):
            # 将方向向量广播到所有采样点
            ray_dirs = np.tile(direction, (len(points), 1))
            
            # 首先快速剔除：如果点本身的法线和加工方向点积 < 0，说明这个面是背对刀具的
            # 在单面加工中，背对刀具的面肯定是“不可加工”的（或者需要翻面）
            # 这里我们放宽条件：只检测“倒扣 (Undercut)”。
            # 也就是：面朝上的点，上方有没有遮挡？
            
            # 计算点积
            dots = np.dot(normals, direction)
            
            # 这里的逻辑是：
            # 我们假设要做“单面加工”或者“双面加工”。
            # 严格筛选：对于一个 Setup，是否存在 Undercut？
            # 简化逻辑：我们检查射向无穷远处的光线是否碰到其他面。
            
            is_occluded = mesh.ray.intersects_any(
                ray_origins = ray_origins,
                ray_directions = ray_dirs
            )
            
            # 计算可见比例
            # 注意：我们只关心那些“法线朝向刀具”的面是否被遮挡。
            # 法线背对刀具的面本身就切不到，属于 Setup 问题，不属于 Undercut。
            
            # 筛选出法线朝上的点 (facing the tool)
            front_facing_indices = np.where(dots > 0)[0]
            
            if len(front_facing_indices) == 0:
                continue
                
            occluded_front_facing = is_occluded[front_facing_indices]
            visible_ratio = 1.0 - (np.sum(occluded_front_facing) / len(front_facing_indices))
            
            if visible_ratio > threshold:
                return True, name

        return False, None

    except Exception as e:
        # STEP 解析失败或网格化失败
        print(f"Error processing {step_file_path}: {e}")
        return False, "Error"

def run_export_stl(model):
    """辅助函数：利用 CQ 导出 STL 供 Trimesh 使用"""
    # 这是一个稍微 hack 的写法，因为 CQ 的 export 默认写文件
    # 我们可以通过临时文件或者 io stream 处理，这里简化逻辑
    temp_name = f"temp_{os.getpid()}.stl"
    cq.exporters.export(model, temp_name)
    with open(temp_name, 'rb') as f:
        data = f.read()
    os.remove(temp_name)
    return data

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你下载了 ABC dataset 的 step 文件夹
    step_folder = "./abc_step_v00" 
    output_list = "cnc_ready_file_list.txt"
    
    step_files = glob.glob(os.path.join(step_folder, "*.step"))[:100] # 先测100个
    
    print(f"Start filtering {len(step_files)} files...")
    
    cnt = 0
    with open(output_list, "w") as f:
        # 可以用 multiprocessing 进一步加速
        for path in step_files:
            is_machinable, direction = check_machinability(path)
            if is_machinable:
                print(f"[KEEP] {path} (Best Dir: {direction})")
                f.write(f"{path},{direction}\n")
                cnt += 1
            else:
                pass
                # print(f"[DROP] {path}")
                
    print(f"Filtering done. Kept {cnt} / {len(step_files)} files.")