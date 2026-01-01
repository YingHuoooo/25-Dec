import torch
import torch.nn as nn
from transformers import AutoModel

class NeuroSymbolicGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        
        # 1. 共享编码器 (Shared Encoder)
        # 负责理解 "Create a box..."
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        
        # 2. 骨架嵌入层
        self.skel_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # 3. 核心解码器 (Transformer Decoder)
        # batch_first=True 非常重要，因为我们的输入是 (Batch, Seq)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 4. 双头输出 (Dual Heads)
        
        # Head A: 结构分类头 (预测下一个词是 box 还是 hole)
        self.cls_head = nn.Linear(hidden_dim, vocab_size)
        
        # Head B: 参数回归头 (预测数值)
        # 当输入是 [ARG] 的隐向量时，输出对应的数值
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 输出标量
        )

    def forward(self, input_ids, attention_mask, skel_ids):
        # A. 编码自然语言
        # enc_output shape: (Batch, Seq_Len, 768)
        enc_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # B. 准备解码器输入
        tgt_emb = self.skel_embed(skel_ids) # (Batch, Seq, 768)
        
        # 生成 Causal Mask (因果掩码)，防止模型看到未来的 Token
        # 也就是预测第 i 个词时，只能看 0 到 i-1 个词
        seq_len = skel_ids.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input_ids.device)
        
        # C. 解码
        # memory_key_padding_mask: 告诉解码器 Encoder 输出中哪些是 PAD，不要关注
        dec_output = self.decoder(
            tgt=tgt_emb, 
            memory=enc_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=(attention_mask == 0)
        )
        
        # D. 输出结果
        logits = self.cls_head(dec_output)     # (Batch, Seq, Vocab_Size)
        values = self.reg_head(dec_output)     # (Batch, Seq, 1)
        
        return logits, values.squeeze(-1)