import tkinter as tk
import cadquery as cq
import trimesh
import os

def run_cad():
    """读取输入 -> 生成模型 -> 弹出3D窗口"""
    # 1. 获取文本框里的代码
    code = text_input.get("1.0", tk.END)
    local_vars = {}
    
    try:
        print("正在计算模型...")
        # 2. 动态执行代码
        exec(code, globals(), local_vars)
        
        # 3. 检查 result 变量
        if 'result' not in local_vars:
            print("错误：代码中没找到 'result' 变量")
            return
            
        # 4. 导出临时 STL 并用 Trimesh 查看
        # 导出 STL
        cq.exporters.export(local_vars['result'], "temp.stl")
        
        # 加载并显示 (这一步会弹出一个独立的 3D 窗口)
        mesh = trimesh.load("temp.stl")
        
        # 这是一个阻塞调用，关闭 3D 窗口后才能再次点击运行
        mesh.show(caption="3D 预览 - 关闭此窗口可继续编辑")
        
    except Exception as e:
        print(f"出错啦: {e}")

# --- 界面布局 (Tkinter) ---
root = tk.Tk()
root.title("简易 CAD 生成器")
root.geometry("600x500")

# 顶部提示
label = tk.Label(root, text="输入代码 (将模型赋值给 result):")
label.pack(pady=5)

# 文本输入框
default_code = """
result = cq.Workplane("XY").box(10, 10, 10).hole(3)
"""
text_input = tk.Text(root, height=20)
text_input.insert(tk.END, default_code.strip())
text_input.pack(fill=tk.BOTH, expand=True, padx=10)

# 运行按钮
btn = tk.Button(root, text="生成 3D 模型", command=run_cad, bg="#DDDDDD", font=("Arial", 12))
btn.pack(pady=10, fill=tk.X, padx=10)

# 启动
root.mainloop()