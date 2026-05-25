"""Task-Conditioned Competence Modeling."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIM = 384
HIDDEN_DIM = 64


class TaskConditionedCompetenceModel(nn.Module):
    """Task embedding + capability matrix (4×12) → competence proxy ``p_hat``, relation ``h_rel``."""

    def __init__(
        self,
        task_dim: int = TASK_DIM,
        hidden_dim: int = HIDDEN_DIM,
        rel_dim: int = 64,
        cross_dropout: float = 0.3,
        proxy_use_bias: bool = True,
        anchor_init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.rel_dim = rel_dim
        self.scale = 1.0 / math.sqrt(float(task_dim))

        self.domain_anchors = nn.Parameter(torch.empty(4, task_dim))
        if anchor_init is not None:
            self.reset_domain_anchors(anchor_init)
        else:
            nn.init.normal_(self.domain_anchors, mean=0.0, std=0.02)

        self.mlp_cap = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )
        self.mlp_diff = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )
        fusion_in = 4 * hidden_dim
        self.mlp_cross = nn.Sequential(
            nn.Linear(fusion_in, rel_dim),
            nn.LayerNorm(rel_dim),
            nn.ReLU(),
            nn.Dropout(p=cross_dropout),
        )
        self.proxy = nn.Linear(rel_dim, 1, bias=proxy_use_bias)

    def reset_domain_anchors(self, embeddings: torch.Tensor) -> None:
        t = embeddings.detach().to(dtype=torch.float32, device=self.domain_anchors.device)
        if t.shape != (4, self.task_dim):
            raise ValueError(
                f"Expected anchor tensor shape (4, {self.task_dim}), got {tuple(t.shape)}"
            )
        self.domain_anchors.data.copy_(t)

    def forward(
        self,
        v_t: torch.Tensor,
        v_m: torch.Tensor,
        return_h_rel: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        scores = (v_t @ self.domain_anchors.T) * self.scale
        w = F.softmax(scores, dim=-1)
        curve = (w.unsqueeze(-1) * v_m).sum(dim=1)
        h_cap = self.mlp_cap(curve)
        h_diff = self.mlp_diff(v_t)
        h_cross = torch.cat(
            [h_cap, h_diff, h_cap - h_diff, h_cap * h_diff],
            dim=-1,
        )
        h_rel = self.mlp_cross(h_cross)
        logits = self.proxy(h_rel)
        p_hat = torch.sigmoid(logits)
        if return_h_rel:
            return p_hat, h_rel
        return p_hat
