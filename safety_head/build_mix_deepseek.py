#!/usr/bin/env python3
"""
DeepSeek-R1-Distill 版：组装 safe/unsafe 平衡训练集 + 训练中未见的 OOD 评估集。
与 build_mix_trainset.py 完全同流程，只是数据来源换成 DeepSeek-8B(draft) 生成、Qwen3Guard 打标的产物。
特征抽取在 70B(base) 上做（见 extract_features.py）。

训练来源（已标注 jsonl，response+label）：
  - output_labeled/DeepSeek-R1-Distill-Llama-8B/{DeepInception,train-00000-of-00001,xstest_prompts,benign_local}_split.jsonl
  - ood_data/jb_train_4methods_ds8b/generated_labeled.jsonl (Attentionshift/CodeChameleon/mousetrap/renellm)
  - mix_data_ds/math_train_labeled.jsonl
OOD（训练完全不含）：
  - ood_data/jb_ood_3methods_ds8b/generated_labeled.jsonl (ABJ/Flip/SoP)
  - mix_data_ds/math_ood_labeled.jsonl
"""
from __future__ import annotations

import argparse
import os

# 复用 build_mix_trainset.py 里的全部逻辑函数
from build_mix_trainset import balance, load_file, report, write_jsonl
import random
from typing import Any, Dict, List, Optional, Tuple

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--total", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_out", default=os.path.join(BASE, "mix_data", "train_mix_deepseek70b.jsonl"))
    ap.add_argument("--ood_out", default=os.path.join(BASE, "mix_data", "ood_eval_deepseek70b.jsonl"))
    ap.add_argument("--ood_total", type=int, default=4000)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    L = os.path.join(BASE, "output_labeled", "DeepSeek-R1-Distill-Llama-8B")
    train_specs: List[Tuple[str, Optional[int]]] = [
        (os.path.join(L, "DeepInception_split.jsonl"), 6000),
        (os.path.join(L, "train-00000-of-00001_split.jsonl"), None),
        (os.path.join(L, "xstest_prompts_split.jsonl"), None),
        (os.path.join(L, "benign_local_split.jsonl"), None),
        (os.path.join(BASE, "ood_data", "jb_train_4methods_ds8b", "generated_labeled.jsonl"), None),
        (os.path.join(BASE, "mix_data_ds", "math_train_labeled.jsonl"), None),
    ]
    ood_specs: List[Tuple[str, Optional[int]]] = [
        (os.path.join(BASE, "ood_data", "jb_ood_3methods_ds8b", "generated_labeled.jsonl"), None),
        (os.path.join(BASE, "mix_data_ds", "math_ood_labeled.jsonl"), None),
    ]

    print("=== Loading TRAIN sources ===")
    train_rows: List[Dict[str, Any]] = []
    for path, cap in train_specs:
        r = load_file(path, cap, rng)
        print(f"  {os.path.basename(path):40s} -> {len(r)}")
        train_rows.extend(r)
    report("TRAIN raw pool", train_rows)
    train_bal = balance(train_rows, args.ratio, args.total, rng)
    report("TRAIN balanced", train_bal)
    write_jsonl(args.train_out, train_bal)

    print("\n=== Loading OOD sources ===")
    ood_rows: List[Dict[str, Any]] = []
    for path, cap in ood_specs:
        r = load_file(path, cap, rng)
        print(f"  {os.path.basename(path):40s} -> {len(r)}")
        ood_rows.extend(r)
    report("OOD raw pool", ood_rows)
    ood_bal = balance(ood_rows, args.ratio, args.ood_total, rng)
    report("OOD balanced", ood_bal)
    write_jsonl(args.ood_out, ood_bal)


if __name__ == "__main__":
    main()
