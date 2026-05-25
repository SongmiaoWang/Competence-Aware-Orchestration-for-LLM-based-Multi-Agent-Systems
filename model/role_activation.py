"""Role Activation — second stage of Competence-Aware Orchestration."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from competence_orchestration.model.task_conditioned_competence import TASK_DIM


def compute_anchor_targets(
    h_T: torch.Tensor,
    R_pool: torch.Tensor,
    eps: float = 1e-6,
    role_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with torch.no_grad():
        h = F.normalize(h_T.detach(), dim=-1, p=2, eps=1e-12)
        if R_pool.dim() == 2:
            r = F.normalize(R_pool.detach(), dim=-1, p=2, eps=1e-12)
            sim = torch.matmul(h, r.T)
            sim_min = sim.min(dim=-1, keepdim=True).values
            sim_max = sim.max(dim=-1, keepdim=True).values
            p = (sim - sim_min) / (sim_max - sim_min + eps)
            return p.detach()

        r = F.normalize(R_pool.detach(), dim=-1, p=2, eps=1e-12)
        sim = torch.bmm(h.unsqueeze(1), r.transpose(1, 2)).squeeze(1)
        if role_valid_mask is None:
            sim_min = sim.min(dim=-1, keepdim=True).values
            sim_max = sim.max(dim=-1, keepdim=True).values
            p = (sim - sim_min) / (sim_max - sim_min + eps)
            return p.detach()

        m = role_valid_mask.float()
        neg_inf = torch.finfo(sim.dtype).min / 4
        sim_masked = sim.masked_fill(m < 0.5, neg_inf)
        sim_max = sim_masked.max(dim=-1, keepdim=True).values
        sim_min_fill = sim.masked_fill(m < 0.5, float("inf"))
        sim_min = sim_min_fill.min(dim=-1, keepdim=True).values
        denom = (sim_max - sim_min).clamp_min(eps)
        p = (sim - sim_min) / denom
        p = p * m
        return p.detach()


def compute_role_activation_losses(
    A: torch.Tensor,
    y: torch.Tensor,
    P_anchor: torch.Tensor,
    lambda_anchor: float = 1.0,
    gamma_scale: float = 0.5,
    role_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_b = y.view(-1)
    if role_valid_mask is None:
        l_anchor = ((A - P_anchor) ** 2).mean()
        mean_a = A.mean(dim=-1)
    else:
        m = role_valid_mask.float()
        diff = (A - P_anchor) ** 2
        denom = m.sum().clamp_min(1.0)
        l_anchor = (diff * m).sum() / denom
        mean_a = (A * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
    l_scale = gamma_scale * ((y_b - 0.5) * mean_a).mean()
    l_total = lambda_anchor * l_anchor + l_scale
    return l_total, l_anchor, l_scale


class RoleActivationModel(nn.Module):
    """Per-role activation from task, role pool, and competence relation vector ``h_rel``."""

    def __init__(
        self,
        d_task: int = TASK_DIM,
        d_competence_rel: int = 64,
        d_hidden: int = 128,
        fuse_hidden: Optional[int] = None,
        fusion: str = "dot_gate",
    ) -> None:
        super().__init__()
        if fusion not in ("dot_gate", "additive"):
            raise ValueError(f"fusion must be 'dot_gate' or 'additive', got {fusion!r}")
        self.d_task = d_task
        self.d_competence_rel = d_competence_rel
        self.d_hidden = d_hidden
        self.fusion = fusion
        fh = fuse_hidden if fuse_hidden is not None else d_hidden
        self.w_r = nn.Linear(d_task, d_hidden)
        self.w_t = nn.Linear(d_task, d_hidden)
        self.ln_fusion = nn.LayerNorm(d_hidden)
        self.role_proj: Optional[nn.Linear]
        if fusion == "dot_gate":
            self.role_proj = nn.Linear(d_task, d_hidden)
        else:
            self.role_proj = None
        self.mlp_fuse = nn.Sequential(
            nn.Linear(d_hidden + d_competence_rel, fh),
            nn.ReLU(),
            nn.Linear(fh, 1),
        )

    def forward(
        self,
        h_T: torch.Tensor,
        R_pool: torch.Tensor,
        h_rel: torch.Tensor,
    ) -> torch.Tensor:
        b = h_T.shape[0]
        if R_pool.dim() == 2:
            m = R_pool.shape[0]
            wr_r = self.w_r(R_pool)
            wt_h = self.w_t(h_T)
            if self.fusion == "dot_gate":
                scale = 1.0 / math.sqrt(float(self.d_hidden))
                scores = torch.matmul(wt_h, wr_r.t()) * scale
                weights = torch.sigmoid(scores)
                gated = weights.unsqueeze(-1) * R_pool.unsqueeze(0)
                assert self.role_proj is not None
                h_prime = self.ln_fusion(torch.relu(self.role_proj(gated)))
            else:
                fused = torch.relu(wr_r.unsqueeze(0) + wt_h.unsqueeze(1))
                h_prime = self.ln_fusion(fused)
            h_rel_e = h_rel.unsqueeze(1).expand(b, m, -1)
            cat = torch.cat([h_prime, h_rel_e], dim=-1)
            logits = self.mlp_fuse(cat).squeeze(-1)
            return torch.sigmoid(logits)

        _b2, m, _d = R_pool.shape
        wr_r = self.w_r(R_pool)
        wt_h = self.w_t(h_T).unsqueeze(1)
        if self.fusion == "dot_gate":
            scale = 1.0 / math.sqrt(float(self.d_hidden))
            scores = (wt_h * wr_r).sum(dim=-1) * scale
            weights = torch.sigmoid(scores)
            gated = weights.unsqueeze(-1) * R_pool
            assert self.role_proj is not None
            h_prime = self.ln_fusion(torch.relu(self.role_proj(gated)))
        else:
            fused = torch.relu(wr_r + wt_h)
            h_prime = self.ln_fusion(fused)
        h_rel_e = h_rel.unsqueeze(1).expand(b, m, -1)
        cat = torch.cat([h_prime, h_rel_e], dim=-1)
        logits = self.mlp_fuse(cat).squeeze(-1)
        return torch.sigmoid(logits)


class RoleActivationPipeline(nn.Module):
    """Task-Conditioned Competence Modeling → Role Activation."""

    def __init__(
        self,
        competence_model: nn.Module,
        role_activation_model: RoleActivationModel,
        random_h_rel: bool = False,
        random_v_t: bool = False,
    ) -> None:
        super().__init__()
        self.competence_model = competence_model
        self.role_activation_model = role_activation_model
        self.random_h_rel = bool(random_h_rel)
        self.random_v_t = bool(random_v_t)

    def forward(
        self,
        v_t: torch.Tensor,
        v_m: torch.Tensor,
        r_pool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if r_pool.dim() not in (2, 3):
            raise ValueError(f"r_pool must be (M,d) or (B,M,d), got {tuple(r_pool.shape)}")
        d_task = self.role_activation_model.d_task
        if r_pool.dim() == 2 and r_pool.shape[1] != d_task:
            raise ValueError(f"r_pool dim {r_pool.shape[1]} != d_task {d_task}")
        if r_pool.dim() == 3 and r_pool.shape[2] != d_task:
            raise ValueError(f"r_pool dim {r_pool.shape[2]} != d_task {d_task}")

        device = v_t.device
        dtype = v_t.dtype
        d_rel = self.role_activation_model.d_competence_rel

        v_eff = torch.randn_like(v_t) if self.random_v_t else v_t
        b = v_eff.shape[0]

        if self.random_h_rel:
            p_hat = torch.full((b, 1), 0.5, device=device, dtype=dtype)
            h_rel = torch.randn(b, d_rel, device=device, dtype=dtype)
        else:
            out = self.competence_model(v_eff, v_m, return_h_rel=True)
            assert isinstance(out, tuple)
            p_hat, h_rel = out
        A = self.role_activation_model(v_eff, r_pool, h_rel)
        return A, p_hat, h_rel, v_eff
