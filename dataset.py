import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer

class CNCDataset(Dataset):
    def __init__(self, data_file, vocab_file, max_len=128):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # 加载骨架词表
        with open(vocab_file, 'r') as f:
            self.skel_vocab = json.load(f)
            
        # 加载预训练模型的 Tokenizer (用于编码自然语言)
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

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
        skel_ids = [self.skel_vocab["[BOS]"]] # Start Token
        for t in skel_tokens:
            skel_ids.append(self.skel_vocab.get(t, 0)) # 0 is PAD/Unknown
        skel_ids.append(self.skel_vocab["[EOS]"]) # End Token
        
        # Padding 骨架
        if len(skel_ids) < self.max_len:
            skel_ids += [0] * (self.max_len - len(skel_ids))
        else:
            skel_ids = skel_ids[:self.max_len]
            
        # 3. 处理参数 (Regression Label)
        # 我们需要知道每个 [ARG] 对应的具体数值
        # 创建一个 "param_values" 序列，长度和 skel_ids 一样
        # 只有在 [ARG] 的位置才有值，其他位置是 0
        param_values = [0.0] * self.max_len
        param_mask = [0] * self.max_len # 用来标记哪些位置需要计算 MSE Loss
        
        p_idx = 0 # 指向 params 列表的指针
        current_params = item['params']
        
        for i, token_id in enumerate(skel_ids):
            if token_id == self.skel_vocab["[ARG]"] and p_idx < len(current_params):
                param_values[i] = float(current_params[p_idx])
                param_mask[i] = 1 # 这个位置需要回归
                p_idx += 1
                
        return {
            "input_ids": text_enc['input_ids'].squeeze(0),
            "attention_mask": text_enc['attention_mask'].squeeze(0),
            "skel_ids": torch.tensor(skel_ids, dtype=torch.long),
            "param_values": torch.tensor(param_values, dtype=torch.float),
            "param_mask": torch.tensor(param_mask, dtype=torch.float) # 1代表这里是[ARG]
        }

    def tokenize_skeleton(self, skel_str):
        for char in ['(', ')', '.', ',', '=']:
            skel_str = skel_str.replace(char, f" {char} ")
        return skel_str.split()