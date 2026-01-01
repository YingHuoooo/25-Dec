import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # 导入 TensorBoard
import os
import json
import time
from tqdm import tqdm

# 导入我们的模块
from dataset import CNCDataset
from model import NeuroSymbolicGenerator
import build_vocab # 导入构建脚本以防万一

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 对于 M1/M2 Mac 用户，可以使用 "mps"
if torch.backends.mps.is_available():
    DEVICE = "mps"

DATA_FILE = "complex_cnc_dataset_qwen2.jsonl" # 你的数据集文件名
VOCAB_FILE = "skeleton_vocab.json"
BATCH_SIZE = 16
EPOCHS = 20         # 建议跑多一点，因为 Loss 下降很快
LR = 5e-5           # 学习率
# =======================================

def main():
    print(f"Using device: {DEVICE}")

    # 1. 检查并构建词表
    if not os.path.exists(VOCAB_FILE):
        print("词表不存在，正在构建...")
        build_vocab.build_skeleton_vocab(DATA_FILE, VOCAB_FILE)
    
    # 加载词表大小
    with open(VOCAB_FILE, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    # 2. 准备数据
    print("加载数据集...")
    dataset = CNCDataset(DATA_FILE, VOCAB_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化模型
    model = NeuroSymbolicGenerator(vocab_size=vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 4. 初始化 Loss 和 TensorBoard
    criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=0) # 0是[PAD]
    criterion_reg = torch.nn.MSELoss(reduction='none')        # 不要在内部求和，我们要手动处理 mask
    
    # 创建 TensorBoard 记录器
    # 运行 `tensorboard --logdir=runs` 查看
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/cnc_experiment_{timestamp}')

    print("🚀 开始训练...")
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # 搬运数据到 GPU/MPS
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            skel_ids = batch['skel_ids'].to(DEVICE)
            param_vals = batch['param_values'].to(DEVICE)
            param_mask = batch['param_mask'].to(DEVICE)
            
            # --- Teacher Forcing 输入输出对齐 ---
            # 我们给模型看 [BOS, A, B]，让它预测 [A, B, EOS]
            # Decoder Input: 去掉最后一个
            dec_input = skel_ids[:, :-1]
            # Target Labels: 去掉第一个
            target_cls = skel_ids[:, 1:]
            
            # 参数也要对应错位
            target_val = param_vals[:, 1:]
            target_mask = param_mask[:, 1:]
            
            # 前向传播
            logits, pred_vals = model(input_ids, mask, dec_input)
            
            # --- 计算 Loss ---
            
            # 1. 结构分类 Loss
            # Flatten: (Batch * Seq, Vocab) vs (Batch * Seq)
            loss_structure = criterion_cls(logits.reshape(-1, vocab_size), target_cls.reshape(-1))
            
            # 2. 参数回归 Loss
            # 先算原始 MSE
            raw_mse = criterion_reg(pred_vals, target_val)
            # 只保留 mask=1 (即[ARG]位置) 的 loss
            masked_mse = raw_mse * target_mask
            # 求平均：总 Error / 有效的参数个数
            # 加 1e-6 防止除以零
            loss_param = masked_mse.sum() / (target_mask.sum() + 1e-6)
            
            # 3. 总 Loss
            # 参数 Loss 权重给大一点，因为回归比分类难
            loss = loss_structure + 2.0 * loss_param
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- TensorBoard 记录 ---
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Structure', loss_structure.item(), global_step)
            writer.add_scalar('Loss/Parameter_MSE', loss_param.item(), global_step)
            
            global_step += 1
            total_loss_epoch += loss.item()
            
            # 进度条显示
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Str': f"{loss_structure.item():.3f}", 
                'Val': f"{loss_param.item():.3f}"
            })

        avg_loss = total_loss_epoch / len(dataloader)
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
        print(f"Epoch {epoch+1} 完成. 平均 Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "neuro_symbolic_cnc_final.pth")
    print("✅ 训练完成！模型已保存。")
    writer.close()

if __name__ == "__main__":
    main()