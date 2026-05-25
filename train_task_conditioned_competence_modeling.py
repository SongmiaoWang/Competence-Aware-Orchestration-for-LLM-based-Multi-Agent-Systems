"""Training — Task-Conditioned Competence Modeling.

Trains ``TaskConditionedCompetenceModel`` with BCE on task-conditioned labels.
Full repo: ``python -m rcm.train --model semantic`` → ``rcm/checkpoints/best.pt``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rcm.domain_anchors import ANCHORS_JSON, ANCHORS_NPY
from rcm.model.rcm_model import DualTowerRCM  # full repo only
from competence_orchestration.model.task_conditioned_competence import TaskConditionedCompetenceModel
from rcm.utils.accuracy_buckets import accuracy_to_bucket
from rcm.utils.dataset import RCMDataset
from rcm.utils.text_encoder import TextEncoder


def _normalize_train_model(name: str) -> str:
    n = name.strip().lower()
    if n in ("dual", "dual_tower"):
        return "dual"
    if n in ("semantic", "semantic_anchored"):
        return "semantic_anchored"
    raise ValueError(f"Unknown --model {name!r}; use dual or semantic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_train_val_split(
    json_path: Path,
    dataset: RCMDataset,
    n_val: int,
    seed: int,
    bucket_from_accuracy: bool = False,
) -> Tuple[Subset, Subset]:
    """Stratified split: each bucket contributes evenly to val; within bucket, random."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    # Group indices by bucket
    by_bucket: dict = defaultdict(list)
    for idx, s in enumerate(samples):
        if bucket_from_accuracy:
            b = accuracy_to_bucket(float(s["accuracy"]))
        else:
            b = int(s["bucket"])
        by_bucket[b].append(idx)

    n_buckets = 4
    base_per_bucket = n_val // n_buckets
    extra = n_val % n_buckets
    targets = [base_per_bucket + (1 if i < extra else 0) for i in range(n_buckets)]

    val_indices: List[int] = []
    for b in range(n_buckets):
        indices = by_bucket.get(b, [])
        n_take = min(targets[b], len(indices))
        rng = random.Random(seed + b)
        val_indices.extend(rng.sample(indices, n_take))

    # If short (some buckets too small), add from buckets with surplus
    need = n_val - len(val_indices)
    if need > 0:
        val_set = set(val_indices)
        for b in range(n_buckets):
            if need <= 0:
                break
            remaining = [i for i in by_bucket.get(b, []) if i not in val_set]
            n_take = min(need, len(remaining))
            if n_take > 0:
                rng = random.Random(seed + 100 + b)
                chosen = rng.sample(remaining, n_take)
                val_indices.extend(chosen)
                val_set.update(chosen)
                need -= n_take

    train_indices = [i for i in range(len(samples)) if i not in set(val_indices)]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _to_bin(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Map values in [0, 1] to bin index 0..n_bins-1. Segment i = [i/n, (i+1)/n)."""
    x = x.clamp(0.0, 1.0)
    b = (x * n_bins).long().clamp(0, n_bins - 1)
    return b


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    acc_n_bins: Tuple[int, ...] = (3, 4, 5, 6),
) -> Tuple[float, float, dict]:
    """Run one training or evaluation epoch.

    Returns:
        (mean_bce_loss, mean_mae, {n_bins: acc})
        acc = fraction where pred and y fall in the same segment (bin) when [0,1] is split into n_bins.
    """
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    n_samples = 0
    correct_by_n_bins: dict = {n: 0 for n in acc_n_bins}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for v_t, v_m, y in loader:
            v_t = v_t.to(device)
            v_m = v_m.to(device)
            y = y.to(device)

            pred = model(v_t, v_m)          # (B, 1)
            loss = criterion(pred, y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            b = pred.shape[0]
            pred_d = pred.detach().squeeze()
            total_loss += loss.item()
            total_mae += (pred_d - y.squeeze()).abs().mean().item()
            n_batches += 1
            n_samples += b

            for n in acc_n_bins:
                pred_bin = _to_bin(pred_d, n)
                y_bin = _to_bin(y.squeeze(), n)
                correct_by_n_bins[n] += (pred_bin == y_bin).sum().item()

    acc_by_n_bins = {n: correct_by_n_bins[n] / n_samples for n in acc_n_bins}
    return total_loss / n_batches, total_mae / n_batches, acc_by_n_bins


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    data: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    val_size: int,
    save_dir: Path,
    seed: int,
    device_str: str,
    encoder_device: str | None,
    embeddings_path: Path | None = None,
    weight_decay: float = 1e-4,
    dropout: float = 0.4,
    patience: int = 5,
    acc_bins: Tuple[int, ...] = (3, 4, 5, 6),
    model: str = "dual",
    anchor_init: Optional[Path] = None,
    cross_dropout: float = 0.3,
    rel_dim: int = 64,
) -> None:
    _set_seed(seed)
    device = torch.device(device_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    emb_path = embeddings_path or (Path(data).parent / "task_embeddings.npy")
    use_cache = Path(emb_path).exists()
    if use_cache:
        print("Loading dataset and task embeddings from cache …")
    else:
        print("Loading dataset and encoding task texts …")

    t0 = time.time()
    encoder = None if use_cache else TextEncoder(device=encoder_device)
    full_dataset = RCMDataset(
        data,
        embeddings_path=embeddings_path,
        encoder=encoder,
        show_encoding_progress=not use_cache,
    )
    print(f"  {len(full_dataset)} samples ready in {time.time() - t0:.1f}s")

    train_set, val_set = _stratified_train_val_split(
        Path(data), full_dataset, n_val=val_size, seed=seed
    )
    print(f"  train={len(train_set)}, val={len(val_set)} (stratified by bucket)")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Model ----
    model_kind = _normalize_train_model(model)
    anchor_path_used: Optional[Path] = None
    if model_kind == "semantic_anchored":
        model = TaskConditionedCompetenceModel(
            rel_dim=rel_dim,
            cross_dropout=cross_dropout,
        ).to(device)
        anchor_path: Optional[Path] = Path(anchor_init) if anchor_init is not None else None
        if anchor_path is not None and not anchor_path.is_file():
            anchor_path = None
        if anchor_path is None and ANCHORS_NPY.is_file():
            anchor_path = ANCHORS_NPY
            print(f"  Using default domain anchors: {anchor_path}")
        if anchor_path is not None and anchor_path.is_file():
            arr = np.load(anchor_path)
            if arr.shape != (4, 384):
                raise ValueError(
                    f"anchor file must be shape (4, 384), got {arr.shape}: {anchor_path}"
                )
            model.reset_domain_anchors(torch.from_numpy(arr.astype(np.float32)))
            anchor_path_used = anchor_path
        else:
            warnings.warn(
                "TaskConditionedCompetenceModel: no anchor .npy found (--anchor_init or "
                f"{ANCHORS_NPY}); domain_anchors are random. Run: python -m rcm.utils.init_domain_anchors",
                stacklevel=2,
            )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TaskConditionedCompetenceModel | trainable params: {n_params:,}")
    else:
        model = DualTowerRCM(dropout=dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"DualTowerRCM | trainable params: {n_params:,}")

    ckpt_args = {
        "model_type": model_kind,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "val_size": val_size,
        "seed": seed,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "patience": patience,
        "cross_dropout": cross_dropout,
        "rel_dim": rel_dim,
        "anchor_init": str(anchor_path_used)
        if anchor_path_used is not None
        else (str(anchor_init) if anchor_init is not None else None),
        "anchors_json": str(ANCHORS_JSON.resolve()) if ANCHORS_JSON.is_file() else None,
    }

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=patience,
    )

    # ---- Training loop ----
    # [0,1] split into n segments; same segment = correct
    best_val_loss = float("inf")
    best_va3 = -1.0
    best_va3_epoch = 0
    prev_lr = lr
    acc_cols = [f"Tr@{n}" for n in acc_bins] + [f"Va@{n}" for n in acc_bins]
    header = (
        f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValMAE':>8}  "
        + "  ".join(f"{c:>6}" for c in acc_cols)
        + f"  {'Time':>6}"
    )
    print("\n" + header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_loss, _, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, acc_n_bins=acc_bins
        )
        val_loss, val_mae, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False, acc_n_bins=acc_bins
        )
        elapsed = time.time() - t_ep

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print(f"  [LR reduced: {prev_lr:.2e} -> {current_lr:.2e}]")
            prev_lr = current_lr

        va3 = val_acc.get(3)
        if va3 is not None:
            improved_best = va3 > best_va3
        else:
            improved_best = val_loss < best_val_loss

        flag = " *" if improved_best else ""
        acc_vals = [train_acc[n] for n in acc_bins] + [val_acc[n] for n in acc_bins]
        acc_str = "  ".join(f"{v:6.2%}" for v in acc_vals)
        print(
            f"{epoch:6d}  {train_loss:10.5f}  {val_loss:10.5f}  {val_mae:8.5f}  "
            f"{acc_str}  {elapsed:5.1f}s{flag}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved_best:
            if va3 is not None:
                best_va3 = va3
                best_va3_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_acc_bins": {str(k): float(v) for k, v in val_acc.items()},
                    "best_metric": "va3" if va3 is not None else "val_loss",
                    "args": dict(ckpt_args),
                },
                save_dir / "best.pt",
            )

    # ---- Final checkpoint ----
    torch.save(
        {
            "epoch": epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "args": dict(ckpt_args),
        },
        save_dir / "last.pt",
    )
    print(f"\nBest val loss: {best_val_loss:.5f}")
    if 3 in acc_bins:
        print(f"Best Va@3: {best_va3:.2%} (epoch {best_va3_epoch}) — saved best.pt by highest Va@3")
    else:
        print("best.pt saved by lowest val_loss (acc_bins does not include 3)")
    print(f"Checkpoints saved to: {save_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RCM: DualTowerRCM or TaskConditionedCompetenceModel"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training_data_collection/training_data_raw/sampled_training_data.json"),
        help="Path to sampled_training_data.json",
    )
    parser.add_argument("--epochs",     type=int,   default=50,     help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Adam learning rate")
    parser.add_argument("--batch_size", type=int,   default=32,     help="Mini-batch size")
    parser.add_argument(
        "--val_size",
        type=int,
        default=70,
        help="Validation set size (stratified by bucket, ~17-18 per bucket)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("rcm/checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--seed",       type=int,   default=42,     help="Global random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model training (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--encoder_device",
        type=str,
        default=None,
        help="Device for the text encoder (default: same as --device)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to task_embeddings.npy (default: <data_dir>/task_embeddings.npy)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Fusion layer dropout (default 0.4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (epochs without val_loss improvement before LR decay)",
    )
    parser.add_argument(
        "--acc_bins",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Segment counts for bin-accuracy (default 3 4 5 6). Same segment = correct.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dual",
        help="Architecture: dual (default) or semantic (TaskConditionedCompetenceModel)",
    )
    parser.add_argument(
        "--anchor_init",
        type=Path,
        default=None,
        help="Optional override for domain_anchors.npy (4,384). If omitted, uses "
        "rcm/domain_anchors/domain_anchors.npy when present. Trained weights are "
        "always in checkpoint state_dict (domain_anchors).",
    )
    parser.add_argument(
        "--cross_dropout",
        type=float,
        default=0.3,
        help="MLP_cross dropout for TaskConditionedCompetenceModel (default 0.3)",
    )
    parser.add_argument(
        "--rel_dim",
        type=int,
        default=64,
        help="h_rel / MLP_cross output dim for TaskConditionedCompetenceModel (default 64)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    enc_device = args.encoder_device or args.device
    train(
        data=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        save_dir=args.save_dir,
        seed=args.seed,
        device_str=args.device,
        encoder_device=enc_device,
        embeddings_path=args.embeddings,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        acc_bins=tuple(args.acc_bins),
        model=args.model,
        anchor_init=args.anchor_init,
        cross_dropout=args.cross_dropout,
        rel_dim=args.rel_dim,
    )
