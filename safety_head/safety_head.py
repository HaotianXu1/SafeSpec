import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class SafetyHeadConfig:
    hidden_size: int
    dropout: float = 0.1
    # ``hidden_states`` 下标，与 HuggingFace 一致：0 为 embedding 后，1..L 为各层输出；``-1`` 表示最后一层。
    hidden_layer_index: int = -1
    # MLP 头：隐层宽度 = max(int(hidden_size * mlp_hidden_ratio), 1)；默认 0.5 即 d→d/2→1
    mlp_hidden_ratio: float = 0.5
    # 隐层个数：1 为 d→mid→1；2 为 d→mid→mid→1；更大则继续在 mid 上堆叠（易过拟合，慎用）
    mlp_num_hidden_layers: int = 1


def resolve_hidden_state_index(layer_index: int, num_states: int) -> int:
    """将用户指定的层下标（可为负，``-1`` 为最后一层）解析为 ``[0, num_states)``。"""
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if layer_index < 0:
        idx = num_states + layer_index
    else:
        idx = layer_index
    if not (0 <= idx < num_states):
        raise ValueError(
            f"hidden_layer_index={layer_index} -> resolved idx={idx}, but model returned "
            f"{num_states} hidden_states (valid range 0..{num_states - 1}, or negative indices)."
        )
    return idx


def select_hidden_state(hidden_states: Sequence[torch.Tensor], layer_index: int) -> torch.Tensor:
    idx = resolve_hidden_state_index(layer_index, len(hidden_states))
    return hidden_states[idx]


class SafetyHead(nn.Module):
    """
    轻量级安全判别头，挂载在目标模型的 hidden state 上。
    输入: [batch, hidden_size]
    输出: [batch, 1]，取 sigmoid 后为 0~1，越大越不安全。

    结构由 ``mlp_hidden_ratio`` 与 ``mlp_num_hidden_layers`` 决定，例如 hidden_size=8192：
    - ratio=0.5, n=1 → 8192→4096→1（默认）
    - ratio=1.0, n=1 → 8192→8192→1
    - ratio=0.25, n=1 → 8192→2048→1
    - ratio=0.5, n=2 → 8192→4096→4096→1
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        mlp_hidden_ratio: float = 0.5,
        mlp_num_hidden_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.mlp_num_hidden_layers = max(1, int(mlp_num_hidden_layers))
        mid = max(int(hidden_size * mlp_hidden_ratio), 1)

        layers: List[nn.Module] = []
        in_f = hidden_size
        for _ in range(self.mlp_num_hidden_layers):
            layers.append(nn.Linear(in_f, mid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_f = mid
        layers.append(nn.Linear(mid, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def save_safety_head(head: SafetyHead, config: SafetyHeadConfig, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": config.__dict__,
        },
        save_path,
    )


def read_safety_head_config(head_path: str) -> SafetyHeadConfig:
    """仅从 checkpoint 读取配置（CPU），用于在加载权重前决定设备等。"""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    return SafetyHeadConfig(
        hidden_size=raw["hidden_size"],
        dropout=raw.get("dropout", 0.1),
        hidden_layer_index=raw.get("hidden_layer_index", -1),
        mlp_hidden_ratio=raw.get("mlp_hidden_ratio", 0.5),
        mlp_num_hidden_layers=raw.get("mlp_num_hidden_layers", 1),
    )


def load_safety_head(
    head_path: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[SafetyHead, SafetyHeadConfig]:
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    raw = ckpt["config"]
    cfg = SafetyHeadConfig(
        hidden_size=raw["hidden_size"],
        dropout=raw.get("dropout", 0.1),
        hidden_layer_index=raw.get("hidden_layer_index", -1),
        mlp_hidden_ratio=raw.get("mlp_hidden_ratio", 0.5),
        mlp_num_hidden_layers=raw.get("mlp_num_hidden_layers", 1),
    )
    head = SafetyHead(
        cfg.hidden_size,
        cfg.dropout,
        mlp_hidden_ratio=cfg.mlp_hidden_ratio,
        mlp_num_hidden_layers=cfg.mlp_num_hidden_layers,
    )
    head.load_state_dict(ckpt["state_dict"])
    # Align safety head weights to target device/dtype to avoid mixed precision issues.
    head = head.to(device=device, dtype=dtype) if dtype is not None else head.to(device)
    head.eval()
    return head, cfg


def pooled_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_len: int = 0,
) -> torch.Tensor:
    """
    对 hidden states 做掩码平均池化，忽略 prefix（如 prompt）和 padding。
    hidden: [bs, seq_len, hidden_size]
    attention_mask: [bs, seq_len]
    prefix_len: 需要从左侧屏蔽的 token 个数（例如 prefix 部分）。
    """
    mask = attention_mask.clone()
    if prefix_len > 0:
        mask[:, :prefix_len] = 0
    mask = mask.unsqueeze(-1)  # [bs, seq_len, 1]
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [bs, 1, 1]
    pooled = (hidden * mask).sum(dim=1, keepdim=True) / denom
    return pooled.squeeze(1)  # [bs, hidden_size]













