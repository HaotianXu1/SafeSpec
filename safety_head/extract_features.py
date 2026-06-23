#!/usr/bin/env python3
"""
用冻结的目标模型 (Qwen3-32B) 对 train/val/OOD 文本**只前向一遍**，抽取并 mask-mean-pool
指定层的 hidden state，缓存到磁盘。之后用 train_head_cached.py 在缓存特征上秒级训练 SafetyHead，
避免每个 epoch / 每次 eval 重复跑 32B（base 冻结 → 特征不变）。

一次可缓存多层（默认最后一层 + 两个中间层），便于对比哪层最可分。

示例:
  python extract_features.py --gpu 4,5,6,7 --batch_size 8 \
    --extract_layers -1,48,32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from glob import glob
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_head import pooled_hidden_states, select_hidden_state


def load_jsonl_samples(path_or_dir: str) -> List[Tuple[str, int]]:
    if os.path.isfile(path_or_dir):
        files = [path_or_dir]
    else:
        files = glob(os.path.join(path_or_dir, "**/*.jsonl"), recursive=True)
    out: List[Tuple[str, int]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = d.get("response", "")
                label = (d.get("label") or "").lower()
                if not text or label not in ("safe", "unsafe"):
                    continue
                out.append((text, 0 if label == "safe" else 1))
    return out


class ListDS(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t, y = self.samples[i]
        return {"text": t, "label": y}


def collate(tokenizer, max_length):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    return _fn


@torch.no_grad()
def extract(base_model, loader, input_device, layers: List[int]) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    feats: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    ys: List[torch.Tensor] = []
    n_done = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, use_cache=False)
        for l in layers:
            hidden = select_hidden_state(outputs.hidden_states, l)
            pooled = pooled_hidden_states(hidden, attention_mask.to(hidden.device), prefix_len=0)
            feats[l].append(pooled.float().cpu())
        ys.append(batch["labels"].clone())
        n_done += input_ids.size(0)
        if n_done % (loader.batch_size * 20) < loader.batch_size:
            print(f"    extracted {n_done} samples")
    X = {l: torch.cat(feats[l], dim=0).half() for l in layers}
    y = torch.cat(ys, dim=0)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-32B")
    ap.add_argument("--train_jsonl", default="/path/to/safety_head_ckpts/mix_data/train_mix_qwen32b.jsonl")
    ap.add_argument("--ood_jsonl", default="/path/to/safety_head_ckpts/mix_data/ood_eval_qwen32b.jsonl")
    ap.add_argument("--out_path", default="/path/to/safety_head_ckpts/mix_data/features_qwen32b.pt")
    ap.add_argument("--extract_layers", default="-1,48,32", help="逗号分隔的 hidden_states 层下标")
    ap.add_argument("--val_ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=3000)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--gpu", default="4,5,6,7")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--shard_id", type=int, default=0, help="数据并行分片号 [0, num_shards)")
    ap.add_argument("--num_shards", type=int, default=1, help="分片总数；>1 时每进程只处理 samples[shard_id::num_shards]")
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--max_memory_gb", type=float, default=0.0,
                    help=">0 时给 device_map=auto 设每卡显存上限(GiB)，迫使权重摊开、给 input_device 留激活余量防 OOM")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = [int(x) for x in args.extract_layers.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.torch_dtype, torch.bfloat16)
    print(f"Loading base model {args.model_path} (frozen feature extractor)...")
    dmap = None if args.device_map in ("", "none", "None") else args.device_map
    max_memory = None
    if args.max_memory_gb > 0 and dmap == "auto":
        # 非对称：input_device(cuda:0) 放更少权重，给前向激活/hidden_states 留余量；其余卡放满。
        n_vis = len([x for x in args.gpu.split(",") if x.strip()])
        max_memory = {0: f"{args.max_memory_gb:g}GiB"}
        for i in range(1, n_vis):
            max_memory[i] = "75GiB"
        print(f"  device_map=auto with max_memory {max_memory}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=dtype,
        device_map=dmap, max_memory=max_memory,
    )
    base_model.eval()
    base_model.requires_grad_(False)
    input_device = next(base_model.parameters()).device
    n_states = base_model.config.num_hidden_layers + 1
    print(f"Model loaded. hidden_states len={n_states}, extracting layers={layers}, input_device={input_device}")

    # 读取并划分 train/val（与最终训练一致的随机划分）
    train_all = load_jsonl_samples(args.train_jsonl)
    random.shuffle(train_all)
    n_val = int(len(train_all) * args.val_ratio)
    val_s = train_all[:n_val]
    train_s = train_all[n_val:]
    ood_s = load_jsonl_samples(args.ood_jsonl)
    print(f"[full] train={len(train_s)} val={len(val_s)} ood={len(ood_s)}")

    # 数据并行：每个进程只处理自己分片（train/val/ood 在所有进程里划分一致，因 seed 相同）
    if args.num_shards > 1:
        train_s = train_s[args.shard_id::args.num_shards]
        val_s = val_s[args.shard_id::args.num_shards]
        ood_s = ood_s[args.shard_id::args.num_shards]
        print(f"[shard {args.shard_id}/{args.num_shards}] train={len(train_s)} val={len(val_s)} ood={len(ood_s)}")

    def make_loader(s):
        return DataLoader(ListDS(s), batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate(tokenizer, args.max_length))

    blob = {"layers": layers, "hidden_size": base_model.config.hidden_size,
            "model_path": args.model_path, "max_length": args.max_length}
    for name, s in [("train", train_s), ("val", val_s), ("ood", ood_s)]:
        print(f"[extract] {name} ({len(s)} samples)...")
        X, y = extract(base_model, make_loader(s), input_device, layers)
        blob[name] = {"X": {str(l): X[l] for l in layers}, "y": y.half()}
        pos = int(y.sum().item())
        print(f"  done {name}: N={y.numel()} unsafe={pos} safe={y.numel()-pos}")

    out_path = args.out_path
    if args.num_shards > 1:
        out_path = f"{args.out_path}.shard{args.shard_id}of{args.num_shards}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(blob, out_path)
    sz = os.path.getsize(out_path) / 1e6
    print(f"Saved features -> {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
