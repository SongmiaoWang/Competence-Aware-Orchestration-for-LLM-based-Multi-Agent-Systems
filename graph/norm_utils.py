from __future__ import annotations

import torch


def min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()
    if (max_val - min_val).item() == 0:
        return tensor * 0.0
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    return normalized_0_to_1 * 2 - 1
