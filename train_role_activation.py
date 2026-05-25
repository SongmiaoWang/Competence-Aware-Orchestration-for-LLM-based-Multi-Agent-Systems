"""Training — Role Activation.

Frozen competence model + ``RoleActivationModel`` via ``RoleActivationPipeline``.
Full repo: ``python -m rcm.train_role_activation`` → ``rcm/checkpoints_role/role_activation_pipeline.pt``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rcm.domain_anchors import ANCHORS_NPY
from competence_orchestration.train.train_task_conditioned_competence_modeling import _stratified_train_val_split
from competence_orchestration.model.role_activation import (
    RoleActivationModel,
    RoleActivationPipeline,
    compute_anchor_targets,
    compute_role_activation_losses,
)
from competence_orchestration.model.task_conditioned_competence import TaskConditionedCompetenceModel
from rcm.utils.role_activation_dataset import RoleActivationDataset
from rcm.utils.text_encoder import TextEncoder
from rcm.utils.dataset_role_pools import (
    max_agents_default,
    pools_config_path,
    role_to_union_index,
    torch_full_union_batch_cpu,
    torch_padded_batch,
)

# Default checkpoint name (distinct from stage-1 best.pt / last.pt).
ROLE_ACTIVATION_CKPT_NAME = "role_activation_pipeline.pt"

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_ROOT.parent.parent / "RCM"


def _load_competence_model_with_anchors(device: torch.device, anchor_init: Optional[Path]) -> TaskConditionedCompetenceModel:
    competence_model = TaskConditionedCompetenceModel().to(device)
    path = Path(anchor_init) if anchor_init is not None else None
    if path is not None and not path.is_file():
        path = None
    if path is None and ANCHORS_NPY.is_file():
        path = ANCHORS_NPY
    if path is not None and path.is_file():
        arr = np.load(path)
        if arr.shape == (4, 384):
            competence_model.reset_domain_anchors(torch.from_numpy(arr.astype(np.float32)).to(device))
    return competence_model


def _accumulate_activation_by_bucket(
    A: torch.Tensor,
    bucket: torch.Tensor,
    thr: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[list, list]:
    """Per-bucket sum of (count of roles with A_i > thr) and sample counts. Buckets 0..3."""
    if valid_mask is not None:
        active = ((A > thr).float() * valid_mask).sum(dim=-1)
    else:
        active = (A > thr).sum(dim=-1).float()
    bucket = bucket.view(-1).long()
    sum_act = [0.0, 0.0, 0.0, 0.0]
    cnt = [0, 0, 0, 0]
    for b in range(4):
        m = bucket == b
        if m.any():
            sum_act[b] += active[m].sum().item()
            cnt[b] += int(m.sum().item())
    return sum_act, cnt


def _merge_bucket_stats(
    total_sum: list,
    total_cnt: list,
    sum_act: list,
    cnt: list,
) -> None:
    for b in range(4):
        total_sum[b] += sum_act[b]
        total_cnt[b] += cnt[b]


def _format_bucket_stats(prefix: str, sum_act: list, cnt: list, thr: float) -> str:
    parts = []
    for b in range(4):
        if cnt[b] > 0:
            mean = sum_act[b] / cnt[b]
            parts.append(f"b{b}={mean:.2f}(n={cnt[b]})")
        else:
            parts.append(f"b{b}=—")
    return f"  [{prefix}] roles with A>{thr:g}: " + " ".join(parts)


def _epoch_losses(
    pipeline: RoleActivationPipeline,
    loader: DataLoader,
    device: torch.device,
    lambda_anchor: float,
    gamma_scale: float,
    train_mode: bool,
    freeze_rcm: bool,
    activation_threshold: float,
    collect_bucket_stats: bool,
) -> Tuple[float, float, float, list, list]:
    """Returns (mean total loss, mean L_anchor, mean L_scale, sum_act×4, cnt×4)."""
    if train_mode:
        pipeline.train()
        if freeze_rcm:
            pipeline.competence_model.eval()
    else:
        pipeline.eval()

    total = 0.0
    sum_la = 0.0
    sum_ls = 0.0
    n_batches = 0
    sum_act = [0.0, 0.0, 0.0, 0.0]
    cnt = [0, 0, 0, 0]
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for v_t, v_m, y, bucket, r_pool, mask in loader:
            v_t = v_t.to(device)
            v_m = v_m.to(device)
            y = y.to(device)
            bucket = bucket.to(device)
            r_pool = r_pool.to(device)
            mask = mask.to(device)
            A, _, _, v_eff = pipeline(v_t, v_m, r_pool)
            P_anchor = compute_anchor_targets(v_eff, r_pool, role_valid_mask=mask)
            loss, l_a, l_s = compute_role_activation_losses(
                A,
                y,
                P_anchor,
                lambda_anchor=lambda_anchor,
                gamma_scale=gamma_scale,
                role_valid_mask=mask,
            )
            total += loss.item()
            sum_la += l_a.item()
            sum_ls += l_s.item()
            n_batches += 1
            if collect_bucket_stats:
                sa, c = _accumulate_activation_by_bucket(
                    A, bucket, activation_threshold, valid_mask=mask
                )
                _merge_bucket_stats(sum_act, cnt, sa, c)
    n = max(n_batches, 1)
    return total / n, sum_la / n, sum_ls / n, sum_act, cnt


def _make_collate_fn(
    union_emb_cpu: torch.Tensor,
    role_to_idx: Dict[str, int],
    max_m: int,
    pools_path: Path,
    role_pool_mode: str,
) -> Callable[[List], Tuple[torch.Tensor, ...]]:
    def collate(batch: List) -> Tuple[torch.Tensor, ...]:
        v_t = torch.stack([b[0] for b in batch], dim=0)
        v_m = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0)
        bucket = torch.stack([b[3] for b in batch], dim=0)
        B = len(batch)
        if role_pool_mode == "full_union":
            r_pool, mask = torch_full_union_batch_cpu(B, union_emb_cpu)
        else:
            ds_list = [b[4] for b in batch]
            r_pool, mask = torch_padded_batch(
                ds_list,
                union_emb_cpu,
                role_to_idx,
                max_m=max_m,
                device=torch.device("cpu"),
                pools_path=pools_path,
            )
        return v_t, v_m, y, bucket, r_pool, mask

    return collate


def _max_roles_in_pools(pools_path: Path) -> int:
    cfg = json.loads(Path(pools_path).read_text(encoding="utf-8"))
    return max(len(spec["roles"]) for spec in cfg["pools"].values())


def train(
    data: Path,
    role_union_embeddings: Path,
    pools_json: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    save_dir: Path,
    seed: int,
    device_str: str,
    embeddings_path: Optional[Path],
    rcm_checkpoint: Optional[Path],
    anchor_init: Optional[Path],
    freeze_rcm: bool,
    lambda_anchor: float,
    gamma_scale: float,
    d_hidden: int,
    weight_decay: float,
    val_size: int,
    activation_threshold: float,
    bucket_stats: bool,
    random_h_rc: bool,
    random_v_t: bool,
    bucket_from_accuracy: bool,
    role_fusion: str,
    role_pool_mode: str,
) -> None:
    torch.manual_seed(seed)
    device = torch.device(device_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    union_path = Path(role_union_embeddings)
    if not union_path.is_file():
        raise FileNotFoundError(
            f"Missing {union_path}. Run: python -m rcm.utils.extract_dataset_role_union_embeddings"
        )
    union_np = np.load(union_path).astype(np.float32)
    union_cpu = torch.from_numpy(union_np)
    m_union = int(union_np.shape[0])
    d_task = int(union_np.shape[1])
    print(f"Role union embeddings: {m_union} roles, dim={d_task}")

    pools_path = Path(pools_json)
    if not pools_path.is_file():
        raise FileNotFoundError(pools_path)
    pools_cfg = json.loads(pools_path.read_text(encoding="utf-8"))
    if role_pool_mode not in ("full_union", "per_dataset"):
        raise ValueError(f"role_pool_mode must be full_union or per_dataset, got {role_pool_mode!r}")
    if role_pool_mode == "full_union":
        max_m = m_union
    else:
        max_m = _max_roles_in_pools(pools_path)
    role_to_idx = role_to_union_index(pools_path)
    print(
        f"Dataset role pools: max_m={max_m}  mode={role_pool_mode}  pools={pools_path}"
    )

    competence_model = _load_competence_model_with_anchors(device, anchor_init)
    if rcm_checkpoint is not None:
        try:
            ck = torch.load(rcm_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(rcm_checkpoint, map_location=device)
        competence_model.load_state_dict(ck["model_state"], strict=True)
        print(f"Loaded competence model weights from {rcm_checkpoint}")

    if freeze_rcm:
        for p in competence_model.parameters():
            p.requires_grad = False
        competence_model.eval()
        print("Competence model frozen (eval mode)")

    if random_h_rc:
        print(
            "Ablation --random_h_rc: competence forward skipped; role_net uses Gaussian h_rel "
            f"(B, {competence_model.rel_dim}) each batch. p_hat is unused (placeholder 0.5)."
        )
    if random_v_t:
        print(
            "Ablation --random_v_t: task embedding replaced by Gaussian noise each batch "
            f"(same shape as v_t, dim={competence_model.task_dim}); competence/role_net/P_anchor use this draw."
        )
    if bucket_from_accuracy:
        print(
            "Buckets from accuracy (not JSON ``bucket``): "
            "[0,0.2) [0.2,0.5) [0.5,0.8] (0.8,1] — see rcm.utils.accuracy_buckets"
        )

    role_net = RoleActivationModel(
        d_task=d_task,
        d_rc=competence_model.rel_dim,
        d_hidden=d_hidden,
        fusion=role_fusion,
    ).to(device)
    print(f"RoleActivationModel fusion={role_fusion}")
    pipeline = RoleActivationPipeline(
        competence_model, role_net, random_h_rc=random_h_rc, random_v_t=random_v_t
    ).to(device)

    emb_path = embeddings_path or (Path(data).parent / "task_embeddings.npy")
    use_cache = Path(emb_path).exists()
    encoder = None if use_cache else TextEncoder(device=device_str)
    full_dataset = RoleActivationDataset(
        data,
        embeddings_path=embeddings_path,
        encoder=encoder,
        show_encoding_progress=not use_cache,
        bucket_from_accuracy=bucket_from_accuracy,
    )
    collate = _make_collate_fn(
        union_cpu, role_to_idx, max_m, pools_path, role_pool_mode
    )

    n_all = len(full_dataset)
    if val_size > 0 and val_size < n_all:
        train_set, val_set = _stratified_train_val_split(
            Path(data),
            full_dataset,
            n_val=val_size,
            seed=seed,
            bucket_from_accuracy=bucket_from_accuracy,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        print(
            f"Samples: {n_all}  train={len(train_set)}  val={len(val_set)}  "
            f"(stratified by bucket, seed={seed})  task_emb_cache={use_cache}"
        )
    else:
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
        )
        val_loader = None
        print(f"Samples: {n_all}  (no val split; use --val_size N)  task_emb_cache={use_cache}")

    params: list[nn.Parameter] = list(pipeline.role_net.parameters())
    if not freeze_rcm and not random_h_rc:
        params += list(pipeline.competence_model.parameters())
    elif not freeze_rcm and random_h_rc:
        print(
            "Note: RCM is not in the forward graph with --random_h_rc; "
            "only role_net is optimized."
        )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    header = (
        f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  "
        f"{'L_anchor_tr':>12}  {'L_scale_tr':>12}  {'L_anchor_va':>12}  {'L_scale_va':>12}  {'Time':>6}"
    )
    print("\n" + header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        pipeline.train()
        if freeze_rcm:
            pipeline.competence_model.eval()

        train_sum = 0.0
        train_la = 0.0
        train_ls = 0.0
        n_batches = 0
        tr_sum_act = [0.0, 0.0, 0.0, 0.0]
        tr_cnt = [0, 0, 0, 0]
        for v_t, v_m, y, bucket, r_pool, mask in train_loader:
            v_t = v_t.to(device)
            v_m = v_m.to(device)
            y = y.to(device)
            if bucket_stats:
                bucket = bucket.to(device)
            r_pool = r_pool.to(device)
            mask = mask.to(device)

            A, _, _, v_eff = pipeline(v_t, v_m, r_pool)
            P_anchor = compute_anchor_targets(v_eff, r_pool, role_valid_mask=mask)
            loss, l_a, l_s = compute_role_activation_losses(
                A,
                y,
                P_anchor,
                lambda_anchor=lambda_anchor,
                gamma_scale=gamma_scale,
                role_valid_mask=mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sum += loss.item()
            train_la += l_a.item()
            train_ls += l_s.item()
            n_batches += 1
            if bucket_stats:
                sa, c = _accumulate_activation_by_bucket(
                    A, bucket, activation_threshold, valid_mask=mask
                )
                _merge_bucket_stats(tr_sum_act, tr_cnt, sa, c)
            if freeze_rcm:
                pipeline.competence_model.eval()

        nb = max(n_batches, 1)
        tr_loss = train_sum / nb
        tr_la = train_la / nb
        tr_ls = train_ls / nb

        if val_loader is not None:
            va_loss, va_la, va_ls, va_sum_act, va_cnt = _epoch_losses(
                pipeline,
                val_loader,
                device,
                lambda_anchor,
                gamma_scale,
                train_mode=False,
                freeze_rcm=freeze_rcm,
                activation_threshold=activation_threshold,
                collect_bucket_stats=bucket_stats,
            )
            va_s = f"{va_loss:10.5f}"
        else:
            va_loss, va_la, va_ls = float("nan"), float("nan"), float("nan")
            va_s = f"{'n/a':>10}"
            va_sum_act, va_cnt = [0.0] * 4, [0] * 4

        print(
            f"{epoch:6d}  {tr_loss:10.5f}  {va_s}  "
            f"{tr_la:12.5f}  {tr_ls:12.5f}  "
            f"{va_la:12.5f}  {va_ls:12.5f}  {time.time() - t0:5.1f}s"
        )
        if bucket_stats:
            print(_format_bucket_stats("train", tr_sum_act, tr_cnt, activation_threshold))
            if val_loader is not None:
                print(_format_bucket_stats("val", va_sum_act, va_cnt, activation_threshold))

    if random_h_rc or random_v_t:
        reasons = []
        if random_h_rc:
            reasons.append("--random_h_rc")
        if random_v_t:
            reasons.append("--random_v_t")
        print(
            f"Skipping checkpoint save ({', '.join(reasons)} ablation; not writing "
            f"{ROLE_ACTIVATION_CKPT_NAME})."
        )
    else:
        out_path = save_dir / ROLE_ACTIVATION_CKPT_NAME
        torch.save(
            {
                "checkpoint_kind": "role_activation_pipeline",
                "pipeline_state": pipeline.state_dict(),
                "role_net_state": pipeline.role_net.state_dict(),
                "competence_model_state": pipeline.competence_model.state_dict(),
                "args": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "freeze_rcm": freeze_rcm,
                    "lambda_anchor": lambda_anchor,
                    "gamma_scale": gamma_scale,
                    "d_hidden": d_hidden,
                    "d_task": d_task,
                    "m_union_roles": m_union,
                    "max_m_roles": max_m,
                    "role_union_embeddings": str(union_path.resolve()),
                    "pools_json": str(pools_path.resolve()),
                    "max_agents_default": max_agents_default(pools_cfg),
                    "val_size": val_size,
                    "activation_threshold": activation_threshold,
                    "bucket_stats": bucket_stats,
                    "random_h_rc": random_h_rc,
                    "random_v_t": random_v_t,
                    "bucket_from_accuracy": bucket_from_accuracy,
                    "role_fusion": role_fusion,
                    "role_pool_mode": role_pool_mode,
                },
            },
            out_path,
        )
        print(
            f"Saved role-activation checkpoint ({ROLE_ACTIVATION_CKPT_NAME}): {out_path.resolve()}\n"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=_RCM_ROOT
        / "training_data_collection"
        / "training_data_raw"
        / "sampled_training_data.json",
    )
    p.add_argument(
        "--role-union-embeddings",
        type=Path,
        default=_REPO_ROOT
        / "training_data_collection"
        / "training_data_raw"
        / "dataset_role_union_embeddings.npy",
        help="Union of all dataset-role-pool embeddings (see extract_dataset_role_union_embeddings).",
    )
    p.add_argument(
        "--pools-json",
        type=Path,
        default=_REPO_ROOT / "rcm" / "agent_profiles" / "dataset_role_pools.json",
        help="Per-dataset ordered role lists.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--save_dir",
        type=Path,
        default=_REPO_ROOT / "rcm" / "checkpoints_role",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--embeddings", type=Path, default=None)
    p.add_argument(
        "--rcm_checkpoint",
        type=Path,
        default=None,
        help="Optional pretrained TaskConditionedCompetenceModel (e.g. rcm/checkpoints/best.pt)",
    )
    p.add_argument("--anchor_init", type=Path, default=None)
    p.add_argument(
        "--freeze_rcm",
        action="store_true",
        help="Train only RoleActivationModel",
    )
    p.add_argument("--lambda_anchor", type=float, default=0.5)
    p.add_argument("--gamma_scale", type=float, default=1.0)
    p.add_argument("--d_hidden", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--val_size",
        type=int,
        default=70,
        help="Stratified val size by metadata bucket (same logic as rcm.train); 0 = train on all data, no val",
    )
    p.add_argument(
        "--activation_threshold",
        type=float,
        default=0.5,
        help="With --bucket_stats: count roles with A_i > this threshold (default 0.5)",
    )
    p.add_argument(
        "--bucket_stats",
        action="store_true",
        help="Each epoch print per-bucket (0–3) mean active-role count on train and val",
    )
    p.add_argument(
        "--random_h_rc",
        action="store_true",
        help="Ablate RCM: skip RCM forward; Gaussian h_rc. Does not save a checkpoint.",
    )
    p.add_argument(
        "--random_v_t",
        action="store_true",
        help="Replace task embedding with Gaussian noise (RCM and P_anchor use same draw). No checkpoint.",
    )
    p.add_argument(
        "--bucket_from_accuracy",
        action="store_true",
        help="Derive bucket 0–3 from sample accuracy (see rcm.utils.accuracy_buckets), not JSON bucket",
    )
    p.add_argument(
        "--role_fusion",
        type=str,
        choices=("dot_gate", "additive"),
        default="dot_gate",
        help="dot_gate: sigmoid(W_r r · W_t h) gates raw role vectors (default). additive: ReLU(W_r r + W_t h)",
    )
    p.add_argument(
        "--role-pool-mode",
        type=str,
        choices=("full_union", "per_dataset"),
        default="full_union",
        help="Training RAN input: full_union = union role table for every sample (default); "
        "per_dataset = slice pool by sample dataset like inference.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data=args.data,
        role_union_embeddings=args.role_union_embeddings,
        pools_json=args.pools_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        seed=args.seed,
        device_str=args.device,
        embeddings_path=args.embeddings,
        rcm_checkpoint=args.rcm_checkpoint,
        anchor_init=args.anchor_init,
        freeze_rcm=args.freeze_rcm,
        lambda_anchor=args.lambda_anchor,
        gamma_scale=args.gamma_scale,
        d_hidden=args.d_hidden,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        activation_threshold=args.activation_threshold,
        bucket_stats=args.bucket_stats,
        random_h_rc=args.random_h_rc,
        random_v_t=args.random_v_t,
        bucket_from_accuracy=args.bucket_from_accuracy,
        role_fusion=args.role_fusion,
        role_pool_mode=args.role_pool_mode,
    )
