"""Graph-based MAS Generation — competence- and role-conditioned spatial edge logits."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

from competence_orchestration.graph.norm_utils import min_max_norm

AblationMode = Literal["full", "no_gate", "no_hrel", "task_role_only"]
# ``no_gate``: h_rel only (no A_i gate). ``no_hrel``: A_i gate only. ``task_role_only``: role + task only.


class GraphBasedMASGenerationHead(nn.Module):
    """Spatial topology head for multi-agent graphs.

    Per-node: ``x_i = concat(role_gated_i, task_embedding, h_rel)`` with
    ``role_gated_i = role_emb_i * (1 + alpha * A_i)`` (ablations zero-out gate / h_rel).

    GCN on ``x`` → MLP node embedding → ``logits_mat = z @ z.T`` → flatten + min-max norm.
    """

    def __init__(
        self,
        role_dim: int = 384,
        rel_dim: int = 64,
        hidden_channels: int = 32,
        embed_dim: int = 32,
        mlp_hidden: int = 64,
        dropout: float = 0.5,
        gate_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.role_dim = int(role_dim)
        self.rel_dim = int(rel_dim)
        self.node_in_dim = self.role_dim + self.role_dim + self.rel_dim
        self.register_buffer("gate_alpha", torch.tensor(float(gate_alpha), dtype=torch.float32))
        self.gcn1 = GCNConv(self.node_in_dim, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.dropout = float(dropout)

    def forward(
        self,
        role_embeddings: torch.Tensor,
        task_embedding: torch.Tensor,
        h_rel: torch.Tensor,
        role_activation: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        ablation: AblationMode = "full",
    ) -> torch.Tensor:
        if role_embeddings.dim() != 2 or role_embeddings.shape[1] != self.role_dim:
            raise ValueError(f"role_embeddings must be (N, {self.role_dim})")
        n = role_embeddings.shape[0]
        t = task_embedding.view(-1, self.role_dim)
        if t.shape[0] == 1:
            t = t.expand(n, -1)
        elif t.shape[0] != n:
            raise ValueError("task_embedding must be (role_dim,) or (N, role_dim)")

        h = h_rel.view(-1, self.rel_dim)
        if h.shape[0] == 1:
            h = h.expand(n, -1)
        elif h.shape[0] != n:
            raise ValueError("h_rel must be (rel_dim,) or (N, rel_dim)")

        if role_activation.shape != (n,):
            raise ValueError(f"role_activation must be (N,), got {tuple(role_activation.shape)}")

        if ablation == "full":
            use_gate, use_hrel = True, True
        elif ablation == "no_gate":
            use_gate, use_hrel = False, True
        elif ablation == "no_hrel":
            use_gate, use_hrel = True, False
        elif ablation == "task_role_only":
            use_gate, use_hrel = False, False
        else:
            raise ValueError(f"unknown ablation: {ablation!r}")

        alpha = self.gate_alpha.to(dtype=role_embeddings.dtype, device=role_embeddings.device)
        if use_gate:
            gated = role_embeddings * (1.0 + alpha * role_activation.unsqueeze(-1))
        else:
            gated = role_embeddings

        if use_hrel:
            h_use = h
        else:
            h_use = torch.zeros_like(h)

        x = torch.cat([gated, t, h_use], dim=-1)
        ei = edge_index.to(device=x.device, dtype=torch.long)
        ei, _ = add_self_loops(ei, num_nodes=n)

        x = F.relu(self.gcn1(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, ei)
        x = self.mlp(x)
        logits_mat = x @ x.t()
        flat = min_max_norm(logits_mat.flatten())
        return flat
