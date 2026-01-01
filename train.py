import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CNCDataset
from model import NeuroSymbolicGenerator
import json
from tqdm import tqdm

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# 1. 准备数据
print("Loading Dataset...")
# 先生成一次词表
# import build_vocab; build_vocab.build_skeleton_vocab("complex_cnc_dataset_qwen.jsonl")

dataset = CNCDataset("complex_cnc_dataset_qwen2.jsonl", "skeleton_vocab.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 加载词表大小
with open("skeleton_vocab.json") as f:
    vocab_size = len(json.load(f))

# 2. 初始化模型
model = NeuroSymbolicGenerator(vocab_size=vocab_size).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# 3. 定义损失函数
criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=0) # 忽略 PAD
criterion_reg = torch.nn.MSELoss(reduction='none')        # 回归损失

print("Start Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader)
    for batch in pbar:
        input_ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        skel_ids = batch['skel_ids'].to(DEVICE)         # [BOS, box, (, [ARG] ...]
        param_vals = batch['param_values'].to(DEVICE)   # [0, 0, 0, 50.0 ...]
        param_mask = batch['param_mask'].to(DEVICE)     # [0, 0, 0, 1 ...]
        
        # Transformer 的输入输出错位 (Teacher Forcing)
        # Input:  [BOS, A, B, C]
        # Target: [A, B, C, EOS]
        dec_input = skel_ids[:, :-1]
        target_cls = skel_ids[:, 1:]
        
        # 对齐参数 Mask 和 Value (也要错位)
        target_val = param_vals[:, 1:]
        target_mask = param_mask[:, 1:]
        
        # 前向传播
        logits, pred_vals = model(input_ids, mask, dec_input)
        
        # --- 计算 Loss (核心创新点) ---
        
        # Loss 1: 结构分类损失 (Cross Entropy)
        # Flatten for loss calculation
        loss_structure = criterion_cls(logits.reshape(-1, vocab_size), target_cls.reshape(-1))
        
        # Loss 2: 参数回归损失 (MSE)
        # 只计算 mask=1 (即 [ARG] 位置) 的损失
        loss_param_raw = criterion_reg(pred_vals, target_val)
        # Apply mask
        loss_param = (loss_param_raw * target_mask).sum() / (target_mask.sum() + 1e-6)
        
        # 总损失：加权求和
        loss = loss_structure + 0.5 * loss_param
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Ep {epoch+1} | Loss: {loss.item():.4f} (Str: {loss_structure.item():.3f}, Val: {loss_param.item():.3f})")
        
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), "neuro_symbolic_cnc.pth")
print("Training Finished!")