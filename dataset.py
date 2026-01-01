import torch
from torch.utils.data import Dataset
import json
import re
from transformers import AutoTokenizer

class CNCDataset(Dataset):
    def __init__(self, data_file, vocab_file, max_len=128):
        self.data = []
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据集文件: {data_file}")
        
        # 加载骨架词表
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.skel_vocab = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到词表文件: {vocab_file}，请先运行 build_vocab.py")
            
        # 加载预训练模型的 Tokenizer (用于编码自然语言)
        # 使用 distilroberta-base，因为它小且快
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def tokenize_skeleton(self, skel_str):
        # 保持和 build_vocab 一致的分词逻辑
        pattern = r"\[ARG\]|[a-zA-Z0-9_]+|[^a-zA-Z0-9_\s]"
        return re.findall(pattern, skel_str)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 处理自然语言 (Encoder Input)
        text_enc = self.text_tokenizer(
            item['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        
        # 2. 处理代码骨架 (Decoder Input / Classification Label)
        skel_tokens = self.tokenize_skeleton(item['skeleton'])
        
        # 构建 ID 序列: [BOS] + tokens + [EOS]
        skel_ids = [self.skel_vocab["[BOS]"]] 
        for t in skel_tokens:
            skel_ids.append(self.skel_vocab.get(t, self.skel_vocab.get("[PAD]")))
        skel_ids.append(self.skel_vocab["[EOS]"]) 
        
        # Padding 骨架
        if len(skel_ids) < self.max_len:
            skel_ids += [self.skel_vocab["[PAD]"]] * (self.max_len - len(skel_ids))
        else:
            skel_ids = skel_ids[:self.max_len]
            
        # 3. 处理参数 (Regression Label)
        # 我们需要创建一个 param_values 向量，只在 [ARG] 的位置填入真实数值
        # 同时创建一个 param_mask，标记哪些位置要算 Loss
        
        param_values = [0.0] * self.max_len
        param_mask = [0.0] * self.max_len 
        
        p_idx = 0 # 指向真实 params 列表的指针
        current_params = item['params']
        
        for i, token_id in enumerate(skel_ids):
            # 如果当前 Token 是 [ARG]，且我们还有参数没填完
            if token_id == self.skel_vocab["[ARG]"] and p_idx < len(current_params):
                # -------------------------------------------------------
                # 【重要技巧】数据归一化
                # 原始尺寸可能是 50.0, 100.0，直接回归会导致 MSE Loss 很大 (几千)
                # 我们这里除以 100.0，把数值压缩到 0~1 左右，训练会更稳定
                # 预测时记得乘回来！
                # -------------------------------------------------------
                val = float(current_params[p_idx]) / 100.0 
                
                param_values[i] = val
                param_mask[i] = 1.0 # 标记：这里需要计算 Regression Loss
                p_idx += 1
                
        return {
            "input_ids": text_enc['input_ids'].squeeze(0),
            "attention_mask": text_enc['attention_mask'].squeeze(0),
            "skel_ids": torch.tensor(skel_ids, dtype=torch.long),
            "param_values": torch.tensor(param_values, dtype=torch.float),
            "param_mask": torch.tensor(param_mask, dtype=torch.float)
        }