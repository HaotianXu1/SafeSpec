#!/usr/bin/env python3
"""
从 /path/to/safespec 本地数据构建良性训练样本：vLLM 生成 → split（与 split.py 相同）→ 默认 label=safe → 抽样后与现有 train jsonl 合并。

本地三目录（与 spec_reason_safe.py 一致）：GSM8K/、gpqa/、hellaswag/。
若需 competition MATH，请自行准备 parquet（含 problem 列），并通过 --math_parquet 与 --sources 中的 math 启用。

示例:
  CUDA_VISIBLE_DEVICES=6,7 python benign_local_pipeline.py \\
    --model_path /path/to/models/Qwen3-4B \\
    --merge_into /path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b.jsonl \\
    --merge_out /path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b_with_benign1k.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# 与 generate.py 中 Qwen3 推理格式一致，但不强调越狱反思（良性任务）
BENIGN_SYSTEM_PROMPT = (
    "You are Qwen3, a helpful reasoning assistant.\n"
    "Always reason step by step inside `</think>` before your final answer."
)


def _try_set_line_buffering() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass


def load_gsm8k_prompts(specroot: str, n: int, rng: random.Random) -> List[Tuple[str, str]]:
    path = os.path.join(specroot, "GSM8K", "train-00000-of-00001.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GSM8K parquet not found: {path}")
    df = pd.read_parquet(path)
    if "question" not in df.columns:
        raise ValueError(f"GSM8K parquet missing 'question' column: {path}")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    out: List[Tuple[str, str]] = []
    for i in idx:
        q = str(df.iloc[i]["question"]).strip()
        instr = (
            "Solve the following math word problem. Show reasoning step by step, then give the final numeric answer.\n\n"
            f"Problem:\n{q}"
        )
        out.append((instr, "gsm8k"))
    return out


def load_gpqa_prompts(specroot: str, n: int, rng: random.Random) -> List[Tuple[str, str]]:
    path = os.path.join(specroot, "gpqa", "gpqa_diamond.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GPQA csv not found: {path}")
    df = pd.read_csv(path)
    need = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"GPQA csv missing column {c!r}: {path}")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    out: List[Tuple[str, str]] = []
    letters = ["A", "B", "C", "D"]
    for i in idx:
        row = df.iloc[i]
        q = str(row["Question"]).strip()
        correct = str(row["Correct Answer"]).strip()
        wrong = [
            str(row["Incorrect Answer 1"]).strip(),
            str(row["Incorrect Answer 2"]).strip(),
            str(row["Incorrect Answer 3"]).strip(),
        ]
        options = [correct] + wrong
        order = list(range(4))
        rng.shuffle(order)
        labeled = [f"{letters[j]}. {options[order[j]]}" for j in range(4)]
        instr = (
            "Answer the following multiple-choice question. Reason step by step, then end with a single line: Answer: X (X is A, B, C, or D).\n\n"
            f"{q}\n\n" + "\n".join(labeled)
        )
        out.append((instr, "gpqa"))
    return out


def load_hellaswag_prompts(specroot: str, n: int, rng: random.Random) -> List[Tuple[str, str]]:
    path = os.path.join(specroot, "hellaswag", "data", "validation-00000-of-00001.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HellaSwag parquet not found: {path}")
    df = pd.read_parquet(path)
    for c in ("ctx", "endings", "label"):
        if c not in df.columns:
            raise ValueError(f"HellaSwag parquet missing {c!r}: {path}")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    letters = ["A", "B", "C", "D"]
    out: List[Tuple[str, str]] = []
    for i in idx:
        row = df.iloc[i]
        ctx = str(row["ctx"]).strip()
        endings = row["endings"]
        if hasattr(endings, "tolist"):
            endings = endings.tolist()
        endings = [str(e).strip() for e in endings]
        if len(endings) != 4:
            continue
        opts = "\n".join(f"{letters[j]}. {endings[j]}" for j in range(4))
        instr = (
            "Choose the most likely continuation. Reason briefly, then end with a line: Answer: X (A/B/C/D).\n\n"
            f"Context: {ctx}\n\n{opts}"
        )
        out.append((instr, "hellaswag"))
    return out


def load_math_prompts(math_parquet: str, n: int, rng: random.Random) -> List[Tuple[str, str]]:
    if not os.path.isfile(math_parquet):
        raise FileNotFoundError(f"--math_parquet not found: {math_parquet}")
    df = pd.read_parquet(math_parquet)
    col = None
    for c in ("problem", "question", "Problem"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"MATH parquet needs a 'problem' or 'question' column: {math_parquet}")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    idx = idx[: min(n, len(idx))]
    out: List[Tuple[str, str]] = []
    for i in idx:
        q = str(df.iloc[i][col]).strip()
        instr = (
            "Solve the following mathematics problem. Show reasoning step by step, then box the final answer using \\boxed{{}} when appropriate.\n\n"
            f"Problem:\n{q}"
        )
        out.append((instr, "math"))
    return out


def collect_prompts(
    specroot: str,
    sources: Sequence[str],
    prompts_per_source: int,
    rng: random.Random,
    math_parquet: Optional[str],
) -> List[Tuple[str, str]]:
    all_p: List[Tuple[str, str]] = []
    for src in sources:
        s = src.strip().lower()
        if s == "gsm8k":
            all_p.extend(load_gsm8k_prompts(specroot, prompts_per_source, rng))
        elif s == "gpqa":
            all_p.extend(load_gpqa_prompts(specroot, prompts_per_source, rng))
        elif s == "hellaswag":
            all_p.extend(load_hellaswag_prompts(specroot, prompts_per_source, rng))
        elif s == "math":
            if not math_parquet:
                raise ValueError("source 'math' requires --math_parquet pointing to a local parquet file.")
            all_p.extend(load_math_prompts(math_parquet, prompts_per_source, rng))
        else:
            raise ValueError(f"Unknown source: {src!r} (use gsm8k, gpqa, hellaswag, math)")
    rng.shuffle(all_p)
    return all_p


def run_vllm_generate(
    model_path: str,
    items: List[Tuple[str, str]],
    system_prompt: str,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    temperature: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM: {model_path} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sp = SamplingParams(max_tokens=max_tokens, n=1, temperature=temperature)

    results: List[Dict[str, Any]] = []
    prompts_text: List[str] = []
    meta_source: List[str] = []

    for user_prompt, src in items:
        full = ""
        if system_prompt:
            full += f"{system_prompt}\n\n"
        full += f"{user_prompt}\n\n</think>"
        prompts_text.append(full)
        meta_source.append(src)

    for start in range(0, len(prompts_text), batch_size):
        chunk_p = prompts_text[start : start + batch_size]
        chunk_s = meta_source[start : start + batch_size]
        chunk_u = [items[start + j][0] for j in range(len(chunk_p))]
        outs = llm.generate(chunk_p, sp)
        for j, output in enumerate(outs):
            sample = output.outputs[0]
            results.append(
                {
                    "original_prompt": chunk_u[j],
                    "full_prompt": chunk_p[j],
                    "generated_text": sample.text,
                    "sample_index": 0,
                    "finish_reason": sample.finish_reason,
                    "benign_source": chunk_s[j],
                }
            )
        print(f"  generated {min(start + batch_size, len(prompts_text))}/{len(prompts_text)}")

    return results


def split_generated_text(
    prompt: str,
    full_prompt: str,
    generated_text: str,
    source_file: str,
    benign_source: str,
) -> List[Dict[str, Any]]:
    steps = [s for s in generated_text.split("\n\n") if s.strip()]
    rows: List[Dict[str, Any]] = []
    current: List[str] = []
    for step in steps:
        current.append(step)
        combined = "\n\n".join(current)
        rows.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt,
                "response": combined,
                "current_step": step,
                "step_index": len(current),
                "source_file": source_file,
                "benign_source": benign_source,
                "label": "safe",
                "safety_categories": [],
            }
        )
    return rows


def stratified_sample(
    rows: List[Dict[str, Any]], k: int, rng: random.Random
) -> List[Dict[str, Any]]:
    if k >= len(rows):
        return rows
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_src.setdefault(r.get("benign_source", "unknown"), []).append(r)
    sources = list(by_src.keys())
    rng.shuffle(sources)
    base = k // len(sources)
    rem = k % len(sources)
    picked: List[Dict[str, Any]] = []
    for i, src in enumerate(sources):
        want = base + (1 if i < rem else 0)
        pool = by_src[src][:]
        rng.shuffle(pool)
        picked.extend(pool[: min(want, len(pool))])
    if len(picked) < k:
        picked_ids = {id(x) for x in picked}
        rest = [r for r in rows if id(r) not in picked_ids]
        rng.shuffle(rest)
        picked.extend(rest[: k - len(picked)])
    rng.shuffle(picked)
    return picked[:k]


def main() -> None:
    _try_set_line_buffering()
    parser = argparse.ArgumentParser(description="Benign local data: generate → split → safe → merge")
    parser.add_argument("--specroot", type=str, default="/path/to/safespec")
    parser.add_argument(
        "--sources",
        type=str,
        default="gsm8k,gpqa,hellaswag",
        help="逗号分隔：gsm8k, gpqa, hellaswag；若含 math 需 --math_parquet",
    )
    parser.add_argument("--math_parquet", type=str, default="", help="本地 MATH parquet（含 problem/question 列）")
    parser.add_argument("--prompts_per_source", type=int, default=120, help="每个来源抽多少道题用于生成")
    parser.add_argument("--final_benign_rows", type=int, default=1000, help="split 后保留的良性总行数（分层抽样）")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", type=str, default="/path/to/models/Qwen3-4B")
    parser.add_argument("--system_prompt", type=str, default=None, help="默认使用 BENIGN_SYSTEM_PROMPT")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=str, default="", help="设置 CUDA_VISIBLE_DEVICES，如 6,7")

    parser.add_argument(
        "--raw_jsonl",
        type=str,
        default="",
        help="原始生成 jsonl 路径（默认 data_gen/raw_output/<model>/generated_benign_local.jsonl）",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="跳过 vLLM，仅对已有 raw_jsonl 做 split/抽样/合并",
    )
    parser.add_argument(
        "--merge_into",
        type=str,
        default="/path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b.jsonl",
    )
    parser.add_argument(
        "--merge_out",
        type=str,
        default="/path/to/safety_head_ckpts/train_data_qwen_4b/train_qwen3_4b_with_benign1k.jsonl",
    )
    parser.add_argument("--no_merge", action="store_true", help="不写合并文件，只输出 split 结果")

    args = parser.parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    rng = random.Random(args.seed)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    math_parquet = args.math_parquet.strip() or None

    model_name = os.path.basename(os.path.normpath(args.model_path))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_raw = os.path.join(base_dir, "data_gen", "raw_output", model_name, "generated_benign_local.jsonl")
    raw_path = args.raw_jsonl.strip() or default_raw

    if not args.skip_generate:
        prompts = collect_prompts(args.specroot, sources, args.prompts_per_source, rng, math_parquet)
        if not prompts:
            raise SystemExit("No prompts collected.")
        print(f"Total prompts: {len(prompts)}")
        sys_prompt = args.system_prompt if args.system_prompt is not None else BENIGN_SYSTEM_PROMPT
        records = run_vllm_generate(
            args.model_path,
            prompts,
            sys_prompt,
            args.max_tokens,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.temperature,
            args.gen_batch_size,
        )
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote raw generations: {raw_path}")
    else:
        if not os.path.isfile(raw_path):
            raise SystemExit(f"--skip_generate but raw jsonl missing: {raw_path}")

    split_rows: List[Dict[str, Any]] = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"skip bad json line {line_idx + 1}")
                continue
            prompt = data.get("original_prompt", "") or data.get("prompt", "")
            full_prompt = data.get("full_prompt", "")
            gen = data.get("generated_text", "")
            benign_src = data.get("benign_source", "unknown")
            if not gen:
                continue
            split_rows.extend(
                split_generated_text(
                    prompt,
                    full_prompt,
                    gen,
                    os.path.basename(raw_path),
                    benign_src,
                )
            )

    print(f"Split rows (before sample): {len(split_rows)}")
    split_rows = stratified_sample(split_rows, args.final_benign_rows, rng)
    print(f"After stratified sample: {len(split_rows)}")

    split_dir = os.path.join(base_dir, "output_split", model_name)
    labeled_dir = os.path.join(base_dir, "output_labeled", model_name)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(labeled_dir, exist_ok=True)
    split_out = os.path.join(split_dir, "benign_local_split.jsonl")
    labeled_out = os.path.join(labeled_dir, "benign_local_split.jsonl")

    with open(split_out, "w", encoding="utf-8") as f:
        for r in split_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(labeled_out, "w", encoding="utf-8") as f:
        for r in split_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote split+labeled(safe): {split_out}\n  and {labeled_out}")

    if args.no_merge:
        return

    merge_into = args.merge_into
    merge_out = args.merge_out
    if not os.path.isfile(merge_into):
        raise SystemExit(f"merge_into not found: {merge_into}")

    os.makedirs(os.path.dirname(merge_out) or ".", exist_ok=True)
    n_old = 0
    with open(merge_into, "r", encoding="utf-8") as fin, open(merge_out, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                n_old += 1
            fout.write(line)
        for r in split_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Merged: {n_old} lines from {merge_into} + {len(split_rows)} benign → {merge_out}")


if __name__ == "__main__":
    main()
