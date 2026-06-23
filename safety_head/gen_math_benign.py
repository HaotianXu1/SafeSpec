#!/usr/bin/env python3
"""
用 draft model (默认 Qwen3-4B) 对本地 MATH 题目生成思维链回答，做 progressive split，
并直接标 label=safe（良性推理，沿用 benign_local_pipeline.py 的约定，无需 guard）。

按题目下标切分 train / ood：前 --n_train 道题用于训练集，最后 --n_ood 道**不同**题目
用于 OOD benign 泛化集，确保训练时未见过这些题。

示例:
  python gen_math_benign.py --gpus 4,5 --tensor_parallel_size 2 \
    --n_train 250 --n_ood 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

BENIGN_SYSTEM_PROMPT = (
    "You are Qwen3, a helpful reasoning assistant.\n"
    "Always reason step by step inside `</think>` before your final answer."
)


def _line_buffer():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass


def load_math(path: str) -> List[str]:
    problems: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("problem") or d.get("question")
            if isinstance(p, str) and p.strip():
                problems.append(p.strip())
    return problems


def build_input(problem: str, system_prompt: str) -> str:
    instr = (
        "Solve the following mathematics problem. Show reasoning step by step, "
        "then box the final answer using \\boxed{} when appropriate.\n\n"
        f"Problem:\n{problem}"
    )
    out = ""
    if system_prompt:
        out += f"{system_prompt}\n\n"
    out += f"{instr}\n\n</think>"
    return out, instr


def progressive_split(prompt: str, full_prompt: str, generated: str, src_tag: str) -> List[Dict[str, Any]]:
    steps = [s for s in generated.split("\n\n") if s.strip()]
    rows: List[Dict[str, Any]] = []
    acc: List[str] = []
    for step in steps:
        acc.append(step)
        rows.append({
            "prompt": prompt,
            "full_prompt": full_prompt,
            "response": "\n\n".join(acc),
            "current_step": step,
            "step_index": len(acc),
            "source_file": src_tag,
            "benign_source": "math",
            "label": "safe",
            "safety_categories": [],
        })
    return rows


def generate(model_path, inputs: List[Tuple[str, str]], max_tokens, tp, gpu_mem, temperature, batch_size):
    from vllm import LLM, SamplingParams
    print(f"Loading vLLM generator: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_mem, trust_remote_code=True)
    sp = SamplingParams(max_tokens=max_tokens, n=1, temperature=temperature)
    full_prompts = [x[0] for x in inputs]
    results = []
    for s in range(0, len(full_prompts), batch_size):
        chunk = full_prompts[s:s + batch_size]
        outs = llm.generate(chunk, sp)
        for j, o in enumerate(outs):
            results.append(o.outputs[0].text)
        print(f"  generated {min(s+batch_size,len(full_prompts))}/{len(full_prompts)}")
    return results


def main():
    _line_buffer()
    ap = argparse.ArgumentParser()
    ap.add_argument("--math_jsonl", default="/path/to/safespec/MATH/test.jsonl")
    ap.add_argument("--model_path", default="Qwen/Qwen3-4B")
    ap.add_argument("--out_dir", default="/path/to/safety_head_ckpts/mix_data")
    ap.add_argument("--n_train", type=int, default=250)
    ap.add_argument("--n_ood", type=int, default=80)
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    problems = load_math(args.math_jsonl)
    print(f"Loaded {len(problems)} MATH problems from {args.math_jsonl}")
    train_probs = problems[: args.n_train]
    ood_probs = problems[len(problems) - args.n_ood:] if args.n_ood > 0 else []
    # 防止 train/ood 重叠
    train_set = set(train_probs)
    ood_probs = [p for p in ood_probs if p not in train_set]
    print(f"train problems={len(train_probs)}, ood problems={len(ood_probs)} (non-overlapping)")

    tagged = [(p, "math_train") for p in train_probs] + [(p, "math_ood") for p in ood_probs]
    inputs = [(build_input(p, BENIGN_SYSTEM_PROMPT)[0], build_input(p, BENIGN_SYSTEM_PROMPT)[1], tag) for p, tag in tagged]
    gen_inputs = [(x[0], x[1]) for x in inputs]

    texts = generate(args.model_path, gen_inputs, args.max_tokens,
                     args.tensor_parallel_size, args.gpu_memory_utilization,
                     args.temperature, args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    train_rows, ood_rows = [], []
    for (full_prompt, instr, tag), gen in zip(inputs, texts):
        if not gen.strip():
            continue
        rows = progressive_split(instr, full_prompt, gen, f"generated_{tag}.jsonl")
        if tag == "math_train":
            train_rows.extend(rows)
        else:
            ood_rows.extend(rows)

    train_out = os.path.join(args.out_dir, "math_train_labeled.jsonl")
    ood_out = os.path.join(args.out_dir, "math_ood_labeled.jsonl")
    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ood_out, "w", encoding="utf-8") as f:
        for r in ood_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote MATH train splits: {train_out} ({len(train_rows)} rows)")
    print(f"Wrote MATH ood   splits: {ood_out} ({len(ood_rows)} rows)")


if __name__ == "__main__":
    main()
