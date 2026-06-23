import argparse
import json
import math
import os
import random
from pathlib import Path


DEFAULT_FILE_RULES = [
    {"pattern": "DeepInception", "sample_rate": 0.2},
]


def load_sampling_config(config_path):
    if not config_path:
        return DEFAULT_FILE_RULES
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise ValueError("sampling_config JSON must contain a list field: rules")
    return rules


def collect_jsonl_files(model_dir, file_names):
    if file_names:
        return [os.path.join(model_dir, name) for name in file_names]
    return sorted(str(p) for p in Path(model_dir).glob("*.jsonl"))


def pick_rule(rules, file_name):
    for rule in rules:
        pattern = rule.get("pattern", "")
        if pattern and pattern in file_name:
            return rule
    return {}


def sample_items(items, sample_rate, max_count, rng):
    sample_rate = 1.0 if sample_rate is None else float(sample_rate)
    if sample_rate < 1.0:
        target = int(len(items) * sample_rate)
        if target < len(items):
            items = rng.sample(items, target)
    if max_count is not None and len(items) > int(max_count):
        items = rng.sample(items, int(max_count))
    return items


def load_labeled_items(file_path, model_root, rules, rng):
    items = []
    rel_path = os.path.relpath(file_path, model_root)
    rule = pick_rule(rules, os.path.basename(file_path))
    sample_rate = rule.get("sample_rate")
    max_count = rule.get("max_count")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(data.get("label", "")).lower()
            if label not in ("safe", "unsafe"):
                continue
            if "source_jsonl" not in data:
                data["source_jsonl"] = rel_path
            items.append(data)

    return sample_items(items, sample_rate, max_count, rng)


def balance_items(items, ratio, total_limit, rng):
    safe_items = [d for d in items if d.get("label") == "safe"]
    unsafe_items = [d for d in items if d.get("label") == "unsafe"]

    if ratio <= 0:
        raise ValueError("safe_unsafe_ratio must be > 0")

    max_safe = min(len(safe_items), int(math.floor(len(unsafe_items) * ratio)))
    max_unsafe = min(len(unsafe_items), int(math.floor(max_safe / ratio)))

    if total_limit:
        total_limit = int(total_limit)
        safe_cap = int(math.floor(total_limit * ratio / (1 + ratio)))
        unsafe_cap = total_limit - safe_cap
        max_safe = min(max_safe, safe_cap)
        max_unsafe = min(max_unsafe, unsafe_cap)

    if max_safe < len(safe_items):
        safe_items = rng.sample(safe_items, max_safe)
    if max_unsafe < len(unsafe_items):
        unsafe_items = rng.sample(unsafe_items, max_unsafe)

    combined = safe_items + unsafe_items
    rng.shuffle(combined)
    return combined, len(safe_items), len(unsafe_items)


def main():
    parser = argparse.ArgumentParser(description="Combine labeled jsonl data with sampling and balancing.")
    parser.add_argument("--target_model", type=str, required=True, help="Model subfolder name in output_labeled")
    parser.add_argument("--files", type=str, default="", help="Comma-separated jsonl file names to include")
    parser.add_argument("--output_name", type=str, default="combined.jsonl", help="Output jsonl filename")
    parser.add_argument("--safe_unsafe_ratio", type=float, default=1.0, help="Target ratio: safe / unsafe")
    parser.add_argument("--total_limit", type=int, default=0, help="Total sample size limit (0 means no limit)")
    parser.add_argument("--sampling_config", type=str, default="", help="Path to sampling config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(base_dir, "output_labeled", args.target_model)
    output_root = os.path.join(base_dir, "output_combined", args.target_model)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Model directory not found: {input_root}")

    file_names = [name.strip() for name in args.files.split(",") if name.strip()]
    jsonl_files = collect_jsonl_files(input_root, file_names)
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found in {input_root}")

    rules = load_sampling_config(args.sampling_config)
    rng = random.Random(args.seed)

    all_items = []
    for file_path in jsonl_files:
        items = load_labeled_items(file_path, input_root, rules, rng)
        all_items.extend(items)

    combined, safe_count, unsafe_count = balance_items(
        all_items,
        ratio=args.safe_unsafe_ratio,
        total_limit=args.total_limit if args.total_limit > 0 else None,
        rng=rng,
    )

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, args.output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Input files: {len(jsonl_files)}")
    print(f"Collected items: {len(all_items)}")
    print(f"Balanced output: {len(combined)} (safe={safe_count}, unsafe={unsafe_count})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
