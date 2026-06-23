#!/usr/bin/env python3
"""
在 extract_features.py 缓存的特征上**秒级**训练 SafetyHead（base 已冻结，特征不变）。
- 每个 epoch 在 val + OOD 上评估（瞬时），监控过拟合 / 泛化；
- best-checkpoint：val 最优阈值 F1 改善即存 safety_head_best.pt，OOD 另存 safety_head_best_ood.pt；
- 可对同一份缓存里多层分别训练，快速对比哪层最可分。

示例:
  python train_head_cached.py --features mix_data/features_qwen32b.pt \
    --layer -1 --save_dir safety_head_ckpt_qwen32b_layerlast_mix --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from safety_head import SafetyHead, SafetyHeadConfig, save_safety_head


def metrics_at_thresholds(probs: torch.Tensor, labels: torch.Tensor, thrs: Sequence[float]):
    rows = []
    best = {"f1": -1.0, "thr": 0.5, "acc": 0.0, "precision": 0.0, "recall": 0.0}
    for t in thrs:
        pred = (probs >= t).long()
        total = labels.numel()
        correct = (pred == labels).sum().item()
        tp = ((pred == 1) & (labels == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0)).sum().item()
        fn = ((pred == 0) & (labels == 1)).sum().item()
        acc = correct / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((t, acc, prec, rec, f1))
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": t, "acc": acc, "precision": prec, "recall": rec}
    return rows, best


@torch.no_grad()
def evaluate(head, X, y, thrs, device):
    head.eval()
    logits = head(X.to(device=device, dtype=next(head.parameters()).dtype)).squeeze(-1)
    probs = torch.sigmoid(logits).float().cpu()
    return metrics_at_thresholds(probs, y.long(), thrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/data/xuhaotian/Safety_head/mix_data/features_qwen32b.pt")
    ap.add_argument("--layer", type=int, default=-1, help="使用缓存中的哪一层 (须在 extract_layers 内)")
    ap.add_argument("--save_dir", default="safety_head_ckpt_qwen32b_layerlast_mix")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mlp_hidden_ratio", type=float, default=0.5)
    ap.add_argument("--mlp_num_hidden_layers", type=int, default=1)
    ap.add_argument("--eval_thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    thrs = [float(x) for x in args.eval_thresholds.split(",") if x.strip()]

    blob = torch.load(args.features, map_location="cpu", weights_only=False)
    lkey = str(args.layer)
    if lkey not in blob["train"]["X"]:
        raise ValueError(f"layer {args.layer} not in cache layers {list(blob['train']['X'].keys())}")
    hidden_size = blob["hidden_size"]

    Xtr = blob["train"]["X"][lkey].float(); ytr = blob["train"]["y"].float()
    Xva = blob["val"]["X"][lkey].float();   yva = blob["val"]["y"].float()
    Xood = blob["ood"]["X"][lkey].float();  yood = blob["ood"]["y"].float()
    print(f"layer={args.layer} | train={Xtr.shape} val={Xva.shape} ood={Xood.shape} | hidden={hidden_size}")
    print(f"train unsafe_rate={ytr.mean():.3f} val={yva.mean():.3f} ood={yood.mean():.3f}")

    head = SafetyHead(hidden_size, args.dropout, mlp_hidden_ratio=args.mlp_hidden_ratio,
                      mlp_num_hidden_layers=args.mlp_num_hidden_layers).to(device)
    cfg = SafetyHeadConfig(hidden_size=hidden_size, dropout=args.dropout,
                           hidden_layer_index=args.layer, mlp_hidden_ratio=args.mlp_hidden_ratio,
                           mlp_num_hidden_layers=args.mlp_num_hidden_layers)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "cached_train_log.csv")
    logf = open(log_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(["epoch", "train_loss", "val_bestF1", "val_thr", "val_acc",
                     "ood_bestF1", "ood_thr", "ood_acc"])

    Xtr_d = Xtr.to(device); ytr_d = ytr.to(device)
    n = Xtr_d.size(0)
    best = {"val": -1.0, "ood": -1.0}
    best_paths = {"val": os.path.join(args.save_dir, "safety_head_best.pt"),
                  "ood": os.path.join(args.save_dir, "safety_head_best_ood.pt")}

    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = Xtr_d[idx]; yb = ytr_d[idx]
            logits = head(xb).squeeze(-1)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / n

        _, vbest = evaluate(head, Xva, yva, thrs, device)
        _, obest = evaluate(head, Xood, yood, thrs, device)
        print(f"[ep {ep:02d}] loss={train_loss:.4f} | VAL f1={vbest['f1']:.4f}@{vbest['thr']:.2f} "
              f"acc={vbest['acc']:.4f} p={vbest['precision']:.3f} r={vbest['recall']:.3f} "
              f"|| OOD f1={obest['f1']:.4f}@{obest['thr']:.2f} acc={obest['acc']:.4f}")
        writer.writerow([ep, f"{train_loss:.5f}", f"{vbest['f1']:.5f}", vbest["thr"], f"{vbest['acc']:.5f}",
                         f"{obest['f1']:.5f}", obest["thr"], f"{obest['acc']:.5f}"])
        logf.flush()

        if vbest["f1"] > best["val"] + 1e-6:
            best["val"] = vbest["f1"]; save_safety_head(head, cfg, best_paths["val"])
            print(f"   [BEST-val] f1={vbest['f1']:.4f} -> {best_paths['val']}")
        if obest["f1"] > best["ood"] + 1e-6:
            best["ood"] = obest["f1"]; save_safety_head(head, cfg, best_paths["ood"])
            print(f"   [BEST-ood] f1={obest['f1']:.4f} -> {best_paths['ood']}")

    final_path = os.path.join(args.save_dir, "safety_head.pt")
    save_safety_head(head, cfg, final_path)
    logf.close()
    print(f"\nSaved final -> {final_path}")
    print(f"[BEST] val F1={best['val']:.4f} ({best_paths['val']}) | ood F1={best['ood']:.4f} ({best_paths['ood']})")
    print(f"Log -> {log_path}")


if __name__ == "__main__":
    main()
