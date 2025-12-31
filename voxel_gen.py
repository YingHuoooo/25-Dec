# voxel_gen.py (简化版思路)
import cadquery as cq
import json
import numpy as np
# 假设你有一个 stl_to_voxel 的库，或者使用 trimesh
# pip install trimesh

def code_to_voxel(code_str, resolution=64):
    # 1. 执行代码，拿到 result 对象
    local_vars = {}
    exec(f"import cadquery as cq\n{code_str}", {}, local_vars)
    result = local_vars['result']
    
    # 2. 导出为 STL
    result.val().exportStl("temp.stl")
    
    # 3. 这里使用伪代码，你需要引入一个体素化函数
    # 推荐使用 trimesh 库: trimesh.voxel.creation.voxelize
    # voxel_grid = trimesh_voxelize("temp.stl", resolution)
    
    # 这是一个占位符，代表 64x64x64 的矩阵
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.int8)
    
    return voxel_grid

# 读取数据集并批量转换
with open("cnc_dataset_train.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        voxels = code_to_voxel(data['code'])
        # 保存为 .npy 文件
        np.save(f"./data/voxels/sample_{data['id']}.npy", voxels)