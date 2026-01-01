import torch
import torch.nn as nn
from transformers import AutoModel

class NeuroSymbolicGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        
        # 1. 共享编码器 (Shared Encoder)
        # 使用 DistilRoBERTa (轻量、快)
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        
        # 2. 骨架解码器 (Structure Branch)
        self.skel_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # 标准的 Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 分类头：预测下一个 Token 是什么 (box? cylinder?)
        self.cls_head = nn.Linear(hidden_dim, vocab_size)
        
        # 3. 参数回归头 (Parameter Branch)
        # 只有当骨架预测为 [ARG] 时，这个头的输出才有效
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 输出一个标量 float
        )

    def forward(self, input_ids, attention_mask, skel_ids):
        # A. 编码自然语言
        enc_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # B. 解码骨架
        # skel_ids 是 Teacher Forcing 的输入 (Shifted Right 在训练循环里处理)
        tgt_emb = self.skel_embed(skel_ids)
        
        # 生成 Mask (防止看到未来)
        tgt_len = skel_ids.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1).to(input_ids.device)
        
        # Transformer 解码
        # memory=enc_output (Cross Attention 到自然语言)
        dec_output = self.decoder(
            tgt=tgt_emb, 
            memory=enc_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(attention_mask == 0)
        )
        
        # C. 双头输出
        logits = self.cls_head(dec_output)   # (Batch, Seq, Vocab)
        values = self.reg_head(dec_output)   # (Batch, Seq, 1)
        
        return logits, values.squeeze(-1)