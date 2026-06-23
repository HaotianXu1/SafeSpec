#!/usr/bin/env python3
"""
组装「适中且 safe/unsafe 平衡」的训练集，以及一个训练中未见过的 OOD 评估集。

训练集来源（已标注 jsonl，response+label）：
  - 已有: output_labeled/Qwen3-4B/{DeepInception,train-00000-of-00001,xstest_prompts,benign_local}_split.jsonl
  - 新越狱: ood_data/jb_train_4methods/generated_labeled.jsonl (Attentionshift/CodeChameleon/mousetrap/renellm)
  - 新MATH: mix_data/math_train_labeled.jsonl
每个来源可设 max_count 上限（随机采样），随后按 safe:unsafe=ratio 平衡并限制总量。

OOD 评估集（训练集完全不含）：
  - ood_data/jb_ood_3methods/generated_labeled.jsonl (ABJ/Flip/SoP)
  - mix_data/math_ood_labeled.jsonl (留出的 MATH 题)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

BASE = os.path.dirname(os.path.abspath(__file__))


def row_src(d: Dict[str, Any], fallback: str) -> str:
    if d.get("method"):
        return f"jb:{d['method']}"
    if d.get("benign_source") and d.get("benign_source") != "-":
        return f"benign:{d['benign_source']}"
    sf = d.get("source_file")
    if sf:
        return sf.replace("generated_", "").replace("_split.jsonl", "").replace(".jsonl", "")
    return fallback


def load_file(path: str, max_count: Optional[int], rng: random.Random) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        print(f"  [warn] missing: {path}")
        return []
    fb = os.path.basename(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(d.get("label", "")).lower()
            resp = d.get("response", "")
            if label not in ("safe", "unsafe") or not resp:
                continue
            rows.append({"response": resp, "label": label, "src": row_src(d, fb)})
    if max_count is not None and len(rows) > max_count:
        rows = rng.sample(rows, max_count)
    return rows


def balance(rows: List[Dict[str, Any]], ratio: float, total: Optional[int], rng: random.Random):
    safe = [r for r in rows if r["label"] == "safe"]
    unsafe = [r for r in rows if r["label"] == "unsafe"]
    max_safe = min(len(safe), int(math.floor(len(unsafe) * ratio)))
    max_unsafe = min(len(unsafe), int(math.floor(max_safe / ratio)))
    if total:
        safe_cap = int(math.floor(total * ratio / (1 + ratio)))
        unsafe_cap = total - safe_cap
        max_safe = min(max_safe, safe_cap)
        max_unsafe = min(max_unsafe, unsafe_cap)
    if max_safe < len(safe):
        safe = rng.sample(safe, max_safe)
    if max_unsafe < len(unsafe):
        unsafe = rng.sample(unsafe, max_unsafe)
    out = safe + unsafe
    rng.shuffle(out)
    return out


def report(name: str, rows: List[Dict[str, Any]]):
    lb = collections.Counter(r["label"] for r in rows)
    src = collections.Counter(r["src"] for r in rows)
    print(f"\n[{name}] total={len(rows)} | safe={lb['safe']} unsafe={lb['unsafe']}")
    for s, c in src.most_common():
        print(f"    {s:28s} {c}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.0, help="safe/unsafe 目标比例")
    ap.add_argument("--total", type=int, default=18000, help="训练集总量上限（适中）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_out", default=os.path.join(BASE, "mix_data", "train_mix_qwen32b.jsonl"))
    ap.add_argument("--ood_out", default=os.path.join(BASE, "mix_data", "ood_eval_qwen32b.jsonl"))
    ap.add_argument("--ood_total", type=int, default=4000, help="OOD 集总量上限")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    L = os.path.join(BASE, "output_labeled", "Qwen3-4B")
    # (path, max_count)  —— 上限用于控制单一来源占比，保持多样性
    train_specs: List[Tuple[str, Optional[int]]] = [
        (os.path.join(L, "DeepInception_split.jsonl"), 6000),         # 主越狱，限量防主导
        (os.path.join(L, "train-00000-of-00001_split.jsonl"), None),  # advbench 直接有害
        (os.path.join(L, "xstest_prompts_split.jsonl"), None),        # XSTest 良性/边界
        (os.path.join(L, "benign_local_split.jsonl"), None),          # gsm8k/gpqa/hellaswag
        (os.path.join(BASE, "ood_data", "jb_train_4methods", "generated_labeled.jsonl"), None),
        (os.path.join(BASE, "mix_data", "math_train_labeled.jsonl"), None),
    ]
    ood_specs: List[Tuple[str, Optional[int]]] = [
        (os.path.join(BASE, "ood_data", "jb_ood_3methods", "generated_labeled.jsonl"), None),
        (os.path.join(BASE, "mix_data", "math_ood_labeled.jsonl"), None),
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
