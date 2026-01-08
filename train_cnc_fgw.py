#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练流水线：意图 (text) → 符号序列 (symbolic) → 度量分布 (metric, mu/sigma) + FGW 对齐
依赖：
  pip install torch transformers pot tqdm tensorboard
用法示例：
  # 预热（关闭 FGW）
  python train_cnc_fgw.py --data complex_cnc_dataset_qwen2-complex.jsonl \
    --text_model bert-base-chinese --batch_size 8 --epochs 2 --lr 2e-4 \
    --w_fgw 0.0 --out_dir ckpt_preheat

  # 继续训练 + 打开 FGW（加载预热权重）
  python train_cnc_fgw.py --data complex_cnc_dataset_qwen2-complex.jsonl \
    --text_model bert-base-chinese --batch_size 8 --epochs 2 --lr 1.5e-4 \
    --w_fgw 0.05 --resume ckpt_preheat/checkpoint_epoch1.pt --out_dir ckpt_fgw
  # TensorBoard
  tensorboard --logdir ckpt_fgw/tb
"""

import os
import json
import math
import random
import argparse
from typing import List, Dict, Any, Tuple, Set

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import ot  # POT: Fused/Generalized Gromov-Wasserstein


# ---------------------------
# 实用工具
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exists(x):
    return x is not None


# ---------------------------
# 符号 tokenizer（数据驱动，稳定 vocab）
# ---------------------------
class SymTokenizer:
    def __init__(self, specials: List[str] = ("<pad>", "<bos>", "<eos>", "<unk>")):
        self.specials = specials
        self.vocab = {s: i for i, s in enumerate(specials)}
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]
        self.frozen = False

    def build_from_dataset(self, jsonl_path: str):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for tok in self.trace_to_tokens(item["symbolic_trace"]):
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab)
        self.frozen = True

    def load_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]
        self.frozen = True

    def __len__(self):
        return len(self.vocab)

    @staticmethod
    def trace_to_tokens(trace: List[Dict[str, Any]]) -> List[str]:
        toks = []
        for op in trace:
            face = op.get("face") or "NONE"
            args = "|".join([f"{k}={v}" for k, v in op["args"].items()])
            toks.append(f"{op['op']}::{face}::{args}")
        return toks

    def encode(self, tokens: List[str]) -> List[int]:
        ids = [self.bos_id]
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                ids.append(self.unk_id)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        inv = {i: t for t, i in self.vocab.items()}
        return [inv.get(i, "<unk>") for i in ids]


# ---------------------------
# 数据集
# ---------------------------
class CncDataset(Dataset):
    def __init__(self, path: str, tokenizer, sym_tok: SymTokenizer, max_len=384):
        self.items = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tokenizer
        self.sym_tok = sym_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def _encode_intent(self, text: str):
        return self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding="max_length",
        )

    def __getitem__(self, idx):
        it = self.items[idx]
        intent = self._encode_intent(it["intent_text"])
        sym_tokens = self.sym_tok.trace_to_tokens(it["symbolic_trace"])
        sym_ids = torch.tensor(self.sym_tok.encode(sym_tokens), dtype=torch.long)
        sym_attn = torch.ones_like(sym_ids)
        metric = {p["name"]: (p["mu"], p["sigma"]) for p in it["metric_params"]}
        explicit = set(it["explicit_constraints"])
        return {
            "intent": intent,
            "sym_ids": sym_ids,
            "sym_attn": sym_attn,
            "metric": metric,
            "explicit": explicit,
        }


def collate_fn(batch, pad_id: int):
    keys = batch[0]["intent"].keys()
    intents = {k: torch.cat([b["intent"][k] for b in batch], dim=0) for k in keys}

    sym_ids = [b["sym_ids"] for b in batch]
    sym_attn = [b["sym_attn"] for b in batch]
    sym_ids = nn.utils.rnn.pad_sequence(sym_ids, batch_first=True, padding_value=pad_id)
    sym_attn = nn.utils.rnn.pad_sequence(sym_attn, batch_first=True, padding_value=0)

    metric_list = [b["metric"] for b in batch]
    explicit_list = [b["explicit"] for b in batch]
    return intents, sym_ids, sym_attn, metric_list, explicit_list


# ---------------------------
# 模型
# ---------------------------
class SymbolicDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.embed(tgt_ids)
        T = tgt_ids.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=tgt_ids.device), diagonal=1).bool()
        dec_out = self.dec(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.lm_head(dec_out)
        return logits, dec_out


class MetricHead(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mu_head = nn.Linear(d_model, 1)
        self.log_sigma_head = nn.Linear(d_model, 1)

    def forward(self, states):
        mu = self.mu_head(states).squeeze(-1)
        log_sigma = self.log_sigma_head(states).squeeze(-1)
        return mu, log_sigma


# ---------------------------
# name → token 位置对齐
# ---------------------------
def parse_token(tok: str):
    parts = tok.split("::")
    op = parts[0] if len(parts) > 0 else ""
    face = parts[1] if len(parts) > 1 else ""
    arg_str = parts[2] if len(parts) > 2 else ""
    args = {}
    if arg_str:
        for kv in arg_str.split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                args[k] = v
    return op, face, args


def name_to_pos(name: str, sym_ids_row: torch.Tensor, id_to_tok: Dict[int, str]) -> int:
    target_op, target_k = name.split(".")
    for i, tid in enumerate(sym_ids_row.tolist()):
        tok = id_to_tok.get(tid, "")
        op, _, args = parse_token(tok)
        if op == target_op and target_k in args:
            return i
    return 0  # fallback


# ---------------------------
# FGW 损失
# ---------------------------
def compute_fgw_loss(text_states: torch.Tensor, sym_states: torch.Tensor, alpha=0.5) -> torch.Tensor:
    losses = []
    for b in range(text_states.size(0)):
        Xt = text_states[b]
        Xs = sym_states[b]

        sim = torch.matmul(Xt, Xs.transpose(0, 1)) / (
            torch.norm(Xt, dim=-1, keepdim=True) @ torch.norm(Xs, dim=-1, keepdim=True).transpose(0, 1) + 1e-6
        )
        M = (1 - sim).clamp(min=0).detach().cpu().numpy()

        Tt, Ts = Xt.size(0), Xs.size(0)
        idx_t = torch.arange(Tt).unsqueeze(1)
        idx_s = torch.arange(Ts).unsqueeze(0)
        C1 = torch.abs(idx_t - idx_t.T).float().cpu().numpy()
        C2 = torch.abs(idx_s - idx_s.T).float().cpu().numpy()

        p = ot.unif(Tt)
        q = ot.unif(Ts)
        _, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=alpha, log=True)
        losses.append(log["fgw_dist"])
    return torch.tensor(losses, device=text_states.device).mean()


# ---------------------------
# 训练循环
# ---------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 构建或加载符号 tokenizer
    sym_tok = SymTokenizer()
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        sym_tok.load_vocab(ckpt["sym_vocab"])
        print(f"[resume] loaded vocab from {args.resume}, size={len(sym_tok)}")
    else:
        sym_tok.build_from_dataset(args.data)
        print(f"[vocab] built from data, size={len(sym_tok)}")
    id_to_tok = {i: t for t, i in sym_tok.vocab.items()}

    # 文本 encoder
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_encoder = AutoModel.from_pretrained(args.text_model).to(device)
    if args.freeze_text_encoder:
        for p in text_encoder.parameters():
            p.requires_grad = False

    # 数据集 / DataLoader
    train_ds = CncDataset(args.data, tokenizer, sym_tok, max_len=args.max_len)
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=sym_tok.pad_id),
    )

    # 模型
    sym_dec = SymbolicDecoder(
        vocab_size=len(sym_tok),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)
    metric_head = MetricHead(d_model=args.d_model).to(device)

    # 优化器
    params = list(sym_dec.parameters()) + list(metric_head.parameters())
    if not args.freeze_text_encoder:
        params += list(text_encoder.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=sym_tok.pad_id)

    # 断点恢复
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sym_dec.load_state_dict(ckpt["sym_dec"])
        metric_head.load_state_dict(ckpt["metric_head"])
        if ckpt.get("text_encoder") is not None and not args.freeze_text_encoder:
            text_encoder.load_state_dict(ckpt["text_encoder"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] from {args.resume}, epoch={start_epoch}, global_step={global_step}")

    # TensorBoard
    tb_dir = os.path.join(args.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # 训练
    for epoch in range(start_epoch, args.epochs):
        sym_dec.train()
        metric_head.train()
        text_encoder.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        for batch in pbar:
            intents, sym_ids, sym_attn, metric_list, explicit_list = batch
            sym_ids = sym_ids.to(device)
            sym_attn = sym_attn.to(device)
            intents = {k: v.to(device) for k, v in intents.items()}

            # encoder
            enc_out = text_encoder(**intents).last_hidden_state
            mem_pad_mask = intents["attention_mask"] == 0

            # decoder (teacher forcing)
            logits, dec_states = sym_dec(
                sym_ids[:, :-1],
                enc_out,
                tgt_key_padding_mask=~sym_attn[:, :-1].bool(),
                memory_key_padding_mask=mem_pad_mask,
            )
            loss_ce = ce_loss_fn(logits.reshape(-1, logits.size(-1)), sym_ids[:, 1:].reshape(-1))

            # metric head
            mu, log_sigma = metric_head(dec_states)
            loss_metric = 0.0
            count_metric = 0
            for b, metric_dict in enumerate(metric_list):
                for name, (y_mu, y_sigma) in metric_dict.items():
                    pos = name_to_pos(name, sym_ids[b], id_to_tok)
                    pred_mu = mu[b, pos]
                    pred_sigma = log_sigma[b, pos].exp().clamp(min=1e-3, max=50.0)
                    y_mu_t = torch.tensor(y_mu, device=device, dtype=torch.float32)
                    if name in explicit_list[b]:
                        loss_metric = loss_metric + torch.nn.functional.l1_loss(pred_mu, y_mu_t)
                    else:
                        loss_metric = loss_metric + 0.5 * (
                            (pred_mu - y_mu_t) ** 2 / (pred_sigma ** 2) + pred_sigma.log()
                        )
                    count_metric += 1
            loss_metric = loss_metric / max(1, count_metric)

            # FGW
            loss_fgw = compute_fgw_loss(enc_out, dec_states, alpha=args.alpha_fgw)

            # 总损失
            loss = args.w_ce * loss_ce + args.w_metric * loss_metric + args.w_fgw * loss_fgw

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()

            global_step += 1
            if global_step % args.log_steps == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ce=f"{loss_ce.item():.4f}",
                    metric=f"{loss_metric.item():.4f}",
                    fgw=f"{loss_fgw.item():.4f}",
                )
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/ce", loss_ce.item(), global_step)
                writer.add_scalar("loss/metric", loss_metric.item(), global_step)
                writer.add_scalar("loss/fgw", loss_fgw.item(), global_step)
                for i, g in enumerate(optim.param_groups):
                    writer.add_scalar(f"lr/group{i}", g["lr"], global_step)

        # 保存 checkpoint
        save_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(
            {
                "sym_dec": sym_dec.state_dict(),
                "metric_head": metric_head.state_dict(),
                "sym_vocab": sym_tok.vocab,
                "text_encoder": text_encoder.state_dict() if not args.freeze_text_encoder else None,
                "optimizer": optim.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "global_step": global_step,
            },
            save_path,
        )
        print(f"[ckpt] saved: {save_path}")
    writer.close()


# ---------------------------
# 推理示例（贪心）
# ---------------------------
@torch.no_grad()
def infer(intent_text: str, sym_tok: SymTokenizer, tokenizer, text_encoder, sym_dec, max_len=64, device="cpu"):
    text_encoder.eval()
    sym_dec.eval()
    enc = tokenizer(intent_text, return_tensors="pt").to(device)
    mem = text_encoder(**enc).last_hidden_state
    mem_pad_mask = enc["attention_mask"] == 0

    generated = [sym_tok.bos_id]
    for _ in range(max_len):
        inp = torch.tensor(generated, device=device).unsqueeze(0)
        logits, _ = sym_dec(inp, mem, memory_key_padding_mask=mem_pad_mask)
        next_id = logits[0, -1].argmax(-1).item()
        generated.append(next_id)
        if next_id == sym_tok.eos_id:
            break
    toks = sym_tok.decode(generated[1:-1])
    return toks


# ---------------------------
# 参数解析
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="JSONL 数据路径")
    ap.add_argument("--text_model", type=str, default="bert-base-chinese")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha_fgw", type=float, default=0.5)
    ap.add_argument("--w_ce", type=float, default=1.0)
    ap.add_argument("--w_metric", type=float, default=1.0)
    ap.add_argument("--w_fgw", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--freeze_text_encoder", action="store_true", help="冻结文本 encoder 以省显存")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint 路径，支持恢复训练和优化器")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)