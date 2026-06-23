"""
在固定数据集上评估已训练的 Safety Head，导出
1) 不同分类阈值下的 precision / recall / F1（CSV，便于画「随阈值变化」曲线）
2) sklearn 风格的 PR 曲线点（precision ~ recall）

用法示例：
  python eval_safety_head_curves.py \\
    --model_path Qwen/Qwen3-32B \\
    --head_path safety_head_ckpt_qwen32b_layer32/safety_head.pt \\
    --data_dir /path/to/safety_head_ckpts/train_data_qwen_4b \\
    --out_dir eval_out_layer32 \\
    --device_map auto --gpu 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from glob import glob
from typing import List, Optional, Tuple


def _early_set_cuda_visible():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    visible = None
    for i, arg in enumerate(sys.argv):
        if arg == "--visible" and i + 1 < len(sys.argv):
            visible = sys.argv[i + 1]
            break
    if visible is None:
        for i, arg in enumerate(sys.argv):
            if arg == "--gpu" and i + 1 < len(sys.argv):
                visible = sys.argv[i + 1]
                break
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible


_early_set_cuda_visible()

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_head import load_safety_head, pooled_hidden_states, read_safety_head_config, select_hidden_state


class SafetyDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.samples: List[Tuple[str, int]] = []
        jsonl_files = glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
        k = max_samples if max_samples else None
        seen = 0
        for fp in jsonl_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = data.get("response", "")
                    label = (data.get("label") or "").lower()
                    y = 0 if label == "safe" else 1
                    if not text:
                        continue
                    if k is None:
                        self.samples.append((text, y))
                    else:
                        seen += 1
                        if len(self.samples) < k:
                            self.samples.append((text, y))
                        else:
                            j = random.randint(0, seen - 1)
                            if j < k:
                                self.samples[j] = (text, y)
        if not self.samples:
            raise ValueError(f"No samples found under {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return {"text": text, "label": label}


class TupleListDataset(Dataset):
    """与 train.ListDataset 相同：显式样本列表。"""

    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return {"text": text, "label": label}


def collate_fn(tokenizer, max_length):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return _fn


def _head_device_from_probe(model, tokenizer, layer_index: int, input_device: torch.device) -> torch.device:
    with torch.no_grad():
        probe = tokenizer(".", return_tensors="pt", add_special_tokens=True)
        probe = {k: v.to(input_device) for k, v in probe.items()}
        out = model(**probe, output_hidden_states=True, use_cache=False)
        h = select_hidden_state(out.hidden_states, layer_index)
        return h.device


def _collect_probs_labels(
    base_model,
    head,
    loader,
    input_device: torch.device,
    head_device: torch.device,
    layer_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probs_list: List[float] = []
    labels_list: List[float] = []
    head.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"]

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = select_hidden_state(outputs.hidden_states, layer_index)
            pooled = pooled_hidden_states(hidden, attention_mask.to(hidden.device), prefix_len=0)
            head_dtype = next(head.parameters()).dtype
            logits = head(pooled.to(device=head_device, dtype=head_dtype)).squeeze(-1)
            p = torch.sigmoid(logits).detach().float().cpu().numpy()
            probs_list.extend(p.tolist())
            labels_list.extend(labels.numpy().tolist())
    return np.asarray(probs_list, dtype=np.float64), np.asarray(labels_list, dtype=np.float64)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> Tuple[float, float, float]:
    pred = (y_score >= t).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def write_threshold_curve_csv(path: str, y_true: np.ndarray, y_score: np.ndarray, n_points: int) -> None:
    thresholds = np.linspace(0.0, 1.0, n_points)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall", "f1"])
        for t in thresholds:
            p, r, f1 = metrics_at_threshold(y_true, y_score, float(t))
            w.writerow([f"{t:.6f}", f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])


def write_pr_curve_csv(path: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    try:
        from sklearn.metrics import precision_recall_curve, auc
    except ImportError:
        print("sklearn 未安装，跳过 pr_curve.csv；仍可查看 metrics_vs_threshold.csv")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = auc(recall, precision)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recall", "precision"])
        for r, p in zip(recall, precision):
            w.writerow([f"{r:.6f}", f"{p:.6f}"])
    summary_path = os.path.join(os.path.dirname(path) or ".", "pr_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"average_precision": float(ap), "positive_rate": float(y_true.mean())}, f, indent=2)
    print(f"PR AUC (average precision) = {ap:.4f}，已写入 {summary_path}")


def maybe_plot(out_dir: str, y_true: np.ndarray, y_score: np.ndarray, n_threshold_points: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过作图。")
        return

    thresholds = np.linspace(0.0, 1.0, n_threshold_points)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        p, r, f1 = metrics_at_threshold(y_true, y_score, float(t))
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precs, label="precision")
    ax.plot(thresholds, recs, label="recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_title("Precision / Recall / F1 vs threshold")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "metrics_vs_threshold.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved {p1}")

    try:
        from sklearn.metrics import precision_recall_curve, auc

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(recall, precision, drawstyle="steps-post")
        ax2.set_xlabel("recall")
        ax2.set_ylabel("precision")
        ax2.set_title(f"PR curve (AP={auc(recall, precision):.4f})")
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        fig2.tight_layout()
        p2 = os.path.join(out_dir, "pr_curve.png")
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        print(f"Saved {p2}")
    except ImportError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Export precision/recall/F1 vs threshold and PR curve for Safety Head")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--head_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="eval_safety_curves", help="输出 CSV/图 的目录")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; use 0 to avoid tokenizer fork issues")
    p.add_argument("--max_length", type=int, default=3000)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch_dtype", type=str, default="auto")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visible", type=str, default="")
    p.add_argument("--gpu", type=str, default="")
    p.add_argument("--max_memory_mb", type=int, default=None)
    p.add_argument(
        "--hidden_layer_index",
        type=int,
        default=None,
        help="覆盖 checkpoint 中的层索引；默认从 head checkpoint 读取",
    )
    p.add_argument(
        "--eval_split",
        type=str,
        choices=["all", "val"],
        default="all",
        help="all=整份数据；val=与 train.py 相同方式打乱后取前 val_ratio 作为验证子集",
    )
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--threshold_points", type=int, default=201, help="0~1 上等间隔阈值个数")
    p.add_argument("--plot", action="store_true", help="若已安装 matplotlib，保存 PNG 曲线图")
    return p.parse_args()


def main():
    args = parse_args()
    if args.visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible
    elif args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_safety_head_config(args.head_path)
    layer_index = cfg.hidden_layer_index if args.hidden_layer_index is None else args.hidden_layer_index

    ds = SafetyDataset(args.data_dir, max_samples=args.max_samples)
    samples = list(ds.samples)
    random.shuffle(samples)
    if args.eval_split == "val" and args.val_ratio > 0:
        n_val = max(1, int(len(samples) * args.val_ratio))
        use_samples = samples[:n_val]
        print(f"eval_split=val: using {len(use_samples)} / {len(samples)} (val_ratio={args.val_ratio}, seed={args.seed})")
    else:
        use_samples = samples
        print(f"eval_split=all: using {len(use_samples)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = DataLoader(
        TupleListDataset(use_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )

    td = args.torch_dtype
    if td == "float16":
        torch_dtype = torch.float16
    elif td == "bfloat16":
        torch_dtype = torch.bfloat16
    elif td == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = torch.float32

    max_memory = None
    if args.max_memory_mb:
        visible_str = args.visible or args.gpu or ""
        gpu_ids = [int(x.strip()) for x in visible_str.split(",") if x.strip() != ""]
        max_memory = {i: f"{args.max_memory_mb}MB" for i in gpu_ids}

    dm = None if args.device_map in ("", "none", "None") else args.device_map
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=dm,
        max_memory=max_memory,
    )
    if dm is None:
        base_model.to(args.device)
    base_model.eval()
    base_model.requires_grad_(False)

    input_device = next(base_model.parameters()).device
    head_device = _head_device_from_probe(base_model, tokenizer, layer_index, input_device)
    head, _ = load_safety_head(args.head_path, head_device, dtype=torch_dtype)

    y_score, y_true = _collect_probs_labels(base_model, head, loader, input_device, head_device, layer_index)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metrics_vs_threshold.csv")
    write_threshold_curve_csv(csv_path, y_true, y_score, args.threshold_points)
    print(f"Wrote {csv_path}")

    pr_path = os.path.join(args.out_dir, "pr_curve.csv")
    write_pr_curve_csv(pr_path, y_true, y_score)

    if args.plot:
        maybe_plot(args.out_dir, y_true, y_score, args.threshold_points)

    # 打印 0.5 阈值与 F1 最优阈值（在扫描网格上）
    p0, r0, f0 = metrics_at_threshold(y_true, y_score, 0.5)
    print(f"At threshold=0.5: precision={p0:.4f}, recall={r0:.4f}, f1={f0:.4f}")

    ts = np.linspace(0.0, 1.0, args.threshold_points)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        _, _, f1 = metrics_at_threshold(y_true, y_score, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    p1, r1, _ = metrics_at_threshold(y_true, y_score, best_t)
    print(f"Best F1 on grid (~{args.threshold_points} pts): threshold={best_t:.4f}, precision={p1:.4f}, recall={r1:.4f}, f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
