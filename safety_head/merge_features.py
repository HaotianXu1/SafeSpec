#!/usr/bin/env python3
"""合并 extract_features.py 的数据并行分片 (*.shard{i}of{N}) 为单一缓存文件。"""
import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", default="/path/to/safety_head_ckpts/mix_data/features_qwen32b.pt")
    ap.add_argument("--num_shards", type=int, default=4)
    args = ap.parse_args()

    shards = sorted(glob.glob(f"{args.out_path}.shard*of{args.num_shards}"))
    if len(shards) != args.num_shards:
        raise SystemExit(f"expected {args.num_shards} shards, found {len(shards)}: {shards}")
    print(f"merging {len(shards)} shards: {[os.path.basename(s) for s in shards]}")

    blobs = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    base = blobs[0]
    layers = base["layers"]
    merged = {"layers": layers, "hidden_size": base["hidden_size"],
              "model_path": base["model_path"], "max_length": base["max_length"]}

    for split in ("train", "val", "ood"):
        Xcat = {str(l): torch.cat([b[split]["X"][str(l)] for b in blobs], dim=0) for l in layers}
        ycat = torch.cat([b[split]["y"] for b in blobs], dim=0)
        merged[split] = {"X": Xcat, "y": ycat}
        pos = int(ycat.sum().item())
        print(f"  {split}: N={ycat.numel()} unsafe={pos} safe={ycat.numel()-pos}")

    torch.save(merged, args.out_path)
    sz = os.path.getsize(args.out_path) / 1e6
    print(f"Saved merged -> {args.out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
