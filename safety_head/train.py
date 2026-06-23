import argparse
import csv
import json
import os
import random
import sys
from glob import glob
from typing import List, Sequence, Tuple

# nohup 重定向到文件时 stdout 常为块缓冲，日志迟迟不刷新；行缓冲便于 tail -f 实时查看
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass


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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_head import (
    SafetyHead,
    SafetyHeadConfig,
    pooled_hidden_states,
    save_safety_head,
    select_hidden_state,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SafetyDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: int = None):
        self.samples: List[Tuple[str, int]] = []
        if os.path.isfile(data_dir) and data_dir.endswith(".jsonl"):
            jsonl_files = [os.path.abspath(data_dir)]
        else:
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
                    y = 0 if label == "safe" else 1  # 0=安全, 1=不安全
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


class ListDataset(Dataset):
    """Wrap a list of (text, label) tuples."""

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


def parse_eval_thresholds(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("--eval_thresholds 不能为空，例如 0.4,0.6,0.8,0.99")
    return out


def run_eval_validation(
    head: nn.Module,
    base_model,
    val_loader: DataLoader,
    val_sample_count: int,
    input_device: torch.device,
    model_device: torch.device,
    hidden_layer_index: int,
    thresholds: Sequence[float],
    epoch: int,
    desc: str,
):
    """
    验证集只做一遍前向，对多个阈值分别统计：
    sigmoid(logit) >= thr 判为 unsafe（正类 label=1）；precision/recall/f1 针对 unsafe。
    """
    head.eval()
    prob_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(model_device)

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = select_hidden_state(outputs.hidden_states, hidden_layer_index)
            pooled = pooled_hidden_states(hidden, attention_mask.to(hidden.device), prefix_len=0)
            logits = head(pooled.to(model_device)).squeeze(-1)
            probs = torch.sigmoid(logits)
            prob_chunks.append(probs.detach())
            label_chunks.append(labels.detach().long())

    all_probs = torch.cat(prob_chunks)
    all_labels = torch.cat(label_chunks)

    print(
        f"[Eval epoch={epoch} {desc}] val_samples={val_sample_count}, pos=unsafe "
        f"(unsafe if p>=thr; metrics per threshold):"
    )
    best_f1 = -1.0
    best_thr = 0.5
    best_acc = 0.0
    for thr in thresholds:
        preds = (all_probs >= thr).long()
        total = all_labels.numel()
        correct = (preds == all_labels).sum().item()
        tp = ((preds == 1) & (all_labels == 1)).sum().item()
        fp = ((preds == 1) & (all_labels == 0)).sum().item()
        fn = ((preds == 0) & (all_labels == 1)).sum().item()

        acc = correct / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(
            f"  thr={thr:.4f}  acc={acc:.4f}  precision={precision:.4f}  "
            f"recall={recall:.4f}  f1={f1:.4f}"
        )
        if f1 > best_f1:
            best_f1, best_thr, best_acc = f1, thr, acc
    head.train()
    return {"best_f1": best_f1, "best_thr": best_thr, "acc_at_best": best_acc}


def train(args):
    if args.visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible
        print(f"Using CUDA_VISIBLE_DEVICES: {args.visible}")
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")

    set_seed(args.seed)
    eval_thresholds = parse_eval_thresholds(args.eval_thresholds)
    init_state_dict = None
    init_cfg = None

    if args.init_head_path:
        ckpt = torch.load(args.init_head_path, map_location="cpu", weights_only=False)
        init_state_dict = ckpt.get("state_dict")
        if init_state_dict is None:
            raise ValueError(f"Invalid checkpoint: missing state_dict in {args.init_head_path}")
        init_cfg = ckpt.get("config", {})
        if not isinstance(init_cfg, dict):
            init_cfg = {}

        # Ensure architecture/layer choices match the checkpoint being resumed.
        for key in ("hidden_layer_index", "dropout", "mlp_hidden_ratio", "mlp_num_hidden_layers"):
            if key in init_cfg and hasattr(args, key):
                cur_val = getattr(args, key)
                init_val = init_cfg[key]
                if cur_val != init_val:
                    print(f"[Resume] Override args.{key}: {cur_val} -> {init_val} (from {args.init_head_path})")
                    setattr(args, key, init_val)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (用于特征提取，不参与训练)...")
    
    # 构建 max_memory 映射（如果指定）
    max_memory = None
    if args.max_memory_mb:
        # args.visible 或 args.gpu 可能是 "0,1,2,3"
        visible_str = args.visible or args.gpu or ""
        gpu_ids = [int(i.strip()) for i in visible_str.split(",") if i.strip() != ""]
        max_memory = {i: f"{args.max_memory_mb}MB" for i in gpu_ids}
        # 如果是 device_map="auto"，第一个 GPU 往往承担更多，我们刻意给它留点空间
        # 这里用户可以通过命令行参数控制，或者我们默认给第一个卡少分点
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_memory=max_memory,
    )
    if args.device_map is None:
        base_model.to(args.device)
    base_model.eval()
    base_model.requires_grad_(False)  # 显式冻结，确保只训练 safety head

    # 基础模型输入所在的设备（通常是第一块卡 / 单卡时的唯一设备）
    input_device = next(base_model.parameters()).device

    # SafetyHead 放在「所选 hidden_states 层」输出所在的设备上，避免跨卡搬运特征
    with torch.no_grad():
        probe = tokenizer(".", return_tensors="pt", add_special_tokens=True)
        probe = {k: v.to(input_device) for k, v in probe.items()}
        probe_out = base_model(
            **probe,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = select_hidden_state(probe_out.hidden_states, args.hidden_layer_index)
        model_device = hidden.device

    n_h = len(probe_out.hidden_states)
    print(
        f"Base model loaded. SafetyHead uses hidden_states[{args.hidden_layer_index}] "
        f"(len={n_h}), placed on {model_device}"
    )
    
    hidden_size = base_model.config.hidden_size
    head = SafetyHead(
        hidden_size,
        args.dropout,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
    )
    head.to(model_device)
    mid = max(int(hidden_size * args.mlp_hidden_ratio), 1)
    print(
        f"SafetyHead MLP: d={hidden_size} → mid={mid} (ratio={args.mlp_hidden_ratio}) "
        f"× {args.mlp_num_hidden_layers} hidden layer(s) → 1"
    )
    if init_state_dict is not None:
        missing, unexpected = head.load_state_dict(init_state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Failed to resume checkpoint {args.init_head_path}: "
                f"missing_keys={missing}, unexpected_keys={unexpected}"
            )
        print(f"[Resume] Loaded safety head weights from {args.init_head_path}")

    dataset = SafetyDataset(args.data_dir, max_samples=args.max_samples)
    all_samples = dataset.samples
    random.shuffle(all_samples)
    n_total = len(all_samples)
    n_val = int(n_total * args.val_ratio) if args.val_ratio > 0 else 0
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    def build_loader(samples, shuffle=False):
        ds = ListDataset(samples)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=collate_fn(tokenizer, args.max_length),
        )

    train_loader = build_loader(train_samples, shuffle=True)
    val_loader = build_loader(val_samples, shuffle=False) if n_val > 0 else None

    # 可选：训练中完全不参与训练的 OOD 集（留出越狱方法 + 未见 benign），用于周期性测泛化
    ood_loader = None
    n_ood = 0
    if args.ood_data_dir:
        ood_ds = SafetyDataset(args.ood_data_dir)
        n_ood = len(ood_ds.samples)
        ood_loader = build_loader(list(ood_ds.samples), shuffle=False)
        print(f"[OOD] loaded {n_ood} held-out samples from {args.ood_data_dir}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "safety_head.pt")
    loss_log_path = os.path.join(args.save_dir, "loss_log.csv")
    step_logs = []

    cfg = SafetyHeadConfig(
        hidden_size=hidden_size,
        dropout=args.dropout,
        hidden_layer_index=args.hidden_layer_index,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
    )
    global_train_step = 0

    # ---- best-checkpoint 机制：验证指标改善时立即额外保存，防止后续过拟合丢失最优解 ----
    # 以 val 的「最优阈值 F1」为主选标准，另对 OOD 单独追踪一份最泛化的权重。
    best_scores = {"val": -1.0, "ood": -1.0}
    best_paths = {
        "val": os.path.join(args.save_dir, "safety_head_best.pt"),
        "ood": os.path.join(args.save_dir, "safety_head_best_ood.pt"),
    }

    def maybe_save_best(kind: str, summary):
        if summary is None:
            return
        f1 = summary["best_f1"]
        if f1 > best_scores[kind] + 1e-6:
            best_scores[kind] = f1
            save_safety_head(head, cfg, best_paths[kind])
            print(
                f"[BEST-{kind}] improved -> F1={f1:.4f} @thr={summary['best_thr']:.3f} "
                f"acc={summary['acc_at_best']:.4f} global_step={global_train_step} "
                f"saved {best_paths[kind]}"
            )

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        running_loss = 0.0
        optimizer.zero_grad()
        n_train_steps = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(model_device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = select_hidden_state(outputs.hidden_states, args.hidden_layer_index)
            
            # 确保 mask 在 hidden 相同的设备上进行池化
            pooled = pooled_hidden_states(hidden, attention_mask.to(hidden.device), prefix_len=0)
            # 显式将 pooled 移动到 head 所在的设备，防止 base_model 输出设备不一致
            logits = head(pooled.to(model_device)).squeeze(-1)
            loss = criterion(logits, labels)

            # 梯度累积
            loss = loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum_steps
            running_loss += loss.item() * args.grad_accum_steps
            step_logs.append((epoch, step, loss.item() * args.grad_accum_steps))

            if args.log_steps and (step % args.log_steps == 0):
                avg_running = running_loss / args.log_steps
                print(f"[Epoch {epoch}] step {step}/{n_train_steps}, loss={avg_running:.4f}")
                running_loss = 0.0

            if (
                args.eval_steps > 0
                and val_loader is not None
                and len(val_samples) > 0
                and step % args.eval_steps == 0
            ):
                val_summary = run_eval_validation(
                    head,
                    base_model,
                    val_loader,
                    len(val_samples),
                    input_device,
                    model_device,
                    args.hidden_layer_index,
                    eval_thresholds,
                    epoch,
                    f"step={step}/{n_train_steps}",
                )
                maybe_save_best("val", val_summary)
                if ood_loader is not None:
                    ood_summary = run_eval_validation(
                        head, base_model, ood_loader, n_ood,
                        input_device, model_device, args.hidden_layer_index,
                        eval_thresholds, epoch, f"OOD step={step}/{n_train_steps}",
                    )
                    maybe_save_best("ood", ood_summary)

            global_train_step += 1
            if args.save_steps > 0 and global_train_step % args.save_steps == 0:
                ckpt_path = os.path.join(
                    args.save_dir, f"safety_head_step_{global_train_step}.pt"
                )
                save_safety_head(head, cfg, ckpt_path)
                print(
                    f"[Checkpoint] epoch={epoch} batch_step={step}/{n_train_steps} "
                    f"global_step={global_train_step} -> {ckpt_path}"
                )

        avg_loss = total_loss / n_train_steps
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f}")

        if val_loader is not None and len(val_samples) > 0:
            # 若已在最后一个 train step 做过 periodic eval，则不再重复
            skip_end = args.eval_steps > 0 and n_train_steps > 0 and (n_train_steps % args.eval_steps == 0)
            if args.eval_steps == 0 or not skip_end:
                val_summary = run_eval_validation(
                    head,
                    base_model,
                    val_loader,
                    len(val_samples),
                    input_device,
                    model_device,
                    args.hidden_layer_index,
                    eval_thresholds,
                    epoch,
                    "end",
                )
                maybe_save_best("val", val_summary)
        if ood_loader is not None:
            ood_summary = run_eval_validation(
                head, base_model, ood_loader, n_ood,
                input_device, model_device, args.hidden_layer_index,
                eval_thresholds, epoch, "OOD end",
            )
            maybe_save_best("ood", ood_summary)

    save_safety_head(head, cfg, save_path)
    print(f"Saved safety head to {save_path}")
    print(
        f"[BEST summary] best val F1={best_scores['val']:.4f} -> {best_paths['val']} | "
        f"best OOD F1={best_scores['ood']:.4f} -> {best_paths['ood']}"
    )

    # 保存 loss 曲线数据
    with open(loss_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss"])
        writer.writerows(step_logs)
    print(f"Saved loss log to {loss_log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a safety head on top of target model hidden states")
    parser.add_argument("--model_path", type=str, default="/data/xuhaotian/model/Qwen3-32B", help="目标模型路径（用于抽取 hidden states）")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/xuhaotian/Safety_head/train_data_qwen_4b",
        help="标注数据目录（递归读 **/*.jsonl），或单个 .jsonl 文件路径",
    )
    parser.add_argument("--save_dir", type=str, default="safety_head_ckpt", help="安全头保存目录")
    parser.add_argument(
        "--init_head_path",
        type=str,
        default="",
        help="可选：从已有 safety_head.pt 继续微调；会自动对齐 hidden_layer_index / MLP 结构相关参数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="每隔多少个全局 train batch step 保存一次 checkpoint（safety_head_step_<N>.pt）；0 表示不中途保存，仅训练结束写 safety_head.pt",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--mlp_hidden_ratio",
        type=float,
        default=0.5,
        help="SafetyHead 隐层宽度 = max(int(d*ratio),1)。0.5≈d/2；1.0=d；0.25=d/4",
    )
    parser.add_argument(
        "--mlp_num_hidden_layers",
        type=int,
        default=1,
        help="隐层段数：1 为 d→mid→1；2 为 d→mid→mid→1；更大易过拟合，可配合更大 dropout",
    )
    parser.add_argument("--max_length", type=int, default=3000)
    parser.add_argument("--max_samples", type=int, default=None, help="调试用，限制样本数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例，0 表示不做验证")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="auto", help="float16/bfloat16/auto")
    parser.add_argument("--device_map", type=str, default="auto", help="None 或 'auto'。32B 建议用 auto 以便切分到多卡。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visible", type=str, default="", help="设置 CUDA_VISIBLE_DEVICES，例如 '0' 或 '0,1'")
    parser.add_argument("--gpu", type=str, default="4,7", help="指定可见 GPU（兼容旧参数），例如 '0' 或 '0,1'")
    parser.add_argument("--max_memory_mb", type=int, default=None, help="限制每个 GPU 用于基础模型的显存(MB)，留出空间给 Head 和 Optimizer")
    parser.add_argument("--log_steps", type=int, default=10, help="每多少个 step 打印一次 batch loss，0 表示不打印")
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=3500,
        help="训练过程中每隔多少个 dataloader step 在验证集上 eval 一次；0 表示仅在每个 epoch 结束时 eval（需 val_ratio>0）",
    )
    parser.add_argument(
        "--eval_thresholds",
        type=str,
        default="0.4,0.6,0.8,0.99",
        help="验证时对多个阈值分别算指标，逗号分隔；sigmoid(logit)>=thr 判为 unsafe",
    )
    parser.add_argument(
        "--ood_data_dir",
        type=str,
        default="",
        help="可选：留出的 OOD 集（jsonl 文件或目录），训练中按 eval_steps 节奏周期性评估泛化，不参与训练",
    )
    parser.add_argument(
        "--hidden_layer_index",
        type=int,
        default=-1,
        help="接在 hidden_states 的哪一层：0 为 embedding 后，1..L 为各 transformer 层后，-1 为最后一层（默认）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

