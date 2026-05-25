"""Training — Graph-based MAS Generation.

Frozen competence + role pipeline; trains ``GraphBasedMASGenerationHead``.
Full repo: ``python -m rcm.train_rcm_card_graph_full`` → ``rcm/checkpoints_graph/rcm_card_edge_head.pt``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_ROOT.parent.parent / "RCM"  # full RCM tree
_SUBMISSION_ROOT = _PKG_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SUBMISSION_ROOT))

from all_datasets.evaluate import get_evaluation_config  # noqa: E402
from all_datasets.loader import SPLIT_TRAIN, SUPPORTED_DATASETS, record_to_input  # noqa: E402
from competence_orchestration.graph.graph_based_mas_generation import GraphBasedMASGenerationHead  # noqa: E402
from graph_mas.masks import flatten_mask, spatial_mask_matrix  # noqa: E402
from mas_runner import MASRunner  # noqa: E402
from rcm.utils.rcm_card_graph_head import load_rcm_card_edge_head_with_meta  # noqa: E402
from run_eval import _build_records_for_eval  # noqa: E402

_ROLE_CKPT = _REPO_ROOT / "rcm" / "checkpoints_role" / "role_activation_pipeline.pt"
_ROLE_PROFILES = _REPO_ROOT / "rcm" / "agent_profiles" / "ofa_profiles.json"
_ROLE_UNION_EMB = (
    _REPO_ROOT / "training_data_collection" / "training_data_raw" / "dataset_role_union_embeddings.npy"
)
_ROLE_POOLS_JSON = _REPO_ROOT / "rcm" / "agent_profiles" / "dataset_role_pools.json"
_VECTORS_ROOT = _REPO_ROOT / "training_data_collection" / "vectors"
_DEFAULT_OUT = _REPO_ROOT / "rcm" / "checkpoints_graph" / "rcm_card_edge_head.pt"
_DEFAULT_TRAINING_Y_JSON = _REPO_ROOT / "training_data_collection" / "training_data_raw" / "sampled_training_data.json"


def _record_for_train_global_index(dataset: str, global_dataset_index: int) -> Any:
    """Resolve one train record by global index (split JSON), excluding anchorset like run_eval."""
    records, gix, _ = _build_records_for_eval(
        dataset, SPLIT_TRAIN, exclude_anchor_samples=True
    )
    for rec, gi in zip(records, gix):
        if int(gi) == int(global_dataset_index):
            return rec
    raise KeyError(
        f"No train row for {dataset=} {global_dataset_index=} "
        f"(after anchor filter; pool has {len(records)} rows)"
    )


def _load_samples_from_manifest(manifest_path: Path, split: str) -> List[Tuple[str, Any, int]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("split") and raw["split"] != split:
        raise SystemExit(
            f"Manifest split {raw['split']!r} != --split {split!r}; regenerate or match split."
        )
    if split != SPLIT_TRAIN:
        raise SystemExit("--manifest is only supported with --split train (train-noanchor pool).")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("Manifest must contain non-empty 'items' list.")
    samples: List[Tuple[str, Any, int]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise SystemExit(f"items[{i}] must be an object.")
        ds = str(it.get("dataset", "")).lower()
        gix = it.get("global_dataset_index")
        if gix is None:
            raise SystemExit(f"items[{i}] missing global_dataset_index.")
        gi = int(gix)
        samples.append((ds, _record_for_train_global_index(ds, gi), gi))
    return samples


def _load_y_accuracy_lookup(paths: List[Union[str, Path]]) -> Dict[Tuple[str, str, str], float]:
    """Map (model_lower, dataset_lower, sample_id) -> accuracy in [0,1] from sampled_training_data-style JSON."""
    out: Dict[Tuple[str, str, str], float] = {}
    for path in paths:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--training-y-json not a file: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        arr = raw.get("samples")
        if not isinstance(arr, list):
            raise SystemExit(f"{p}: expected top-level 'samples' list.")
        for j, s in enumerate(arr):
            if not isinstance(s, dict):
                continue
            m = str(s.get("model", "")).strip().lower()
            ds = str(s.get("dataset", "")).strip().lower()
            sid = str(s.get("sample_id", "")).strip()
            acc = s.get("accuracy")
            if not m or not ds or not sid:
                continue
            if acc is None:
                continue
            try:
                y = float(acc)
            except (TypeError, ValueError):
                continue
            y = max(0.0, min(1.0, y))
            out[(m, ds, sid)] = y
    return out


def _sample_id_for_y_lookup(dataset: str, global_dataset_index: int) -> str:
    return f"{str(dataset).lower()}_{int(global_dataset_index)}"


def _lookup_training_y(
    y_lookup: Optional[Dict[Tuple[str, str, str], float]],
    vec_key: str,
    dataset: str,
    global_dataset_index: int,
) -> Optional[float]:
    if not y_lookup:
        return None
    sid = _sample_id_for_y_lookup(dataset, global_dataset_index)
    mk = str(vec_key).strip().lower()
    k = (mk, str(dataset).lower(), sid)
    if k in y_lookup:
        return y_lookup[k]
    # tolerate exact-case model string in JSON keys
    k2 = (str(vec_key).strip(), str(dataset).lower(), sid)
    return y_lookup.get(k2)


class _MockLLM:
    """Minimal LLM stub for gradient / plumbing checks (not for real accuracy)."""

    def gen(self, messages: List[Dict[str, str]]) -> str:  # noqa: ARG002
        return "answer_stub"


def _openrouter_slug(vectors_key: str) -> str:
    import re  # noqa: PLC0415

    if not vectors_key or "/" in vectors_key:
        return vectors_key
    if re.match(r"^(gpt-|o[0-9]+|chatgpt-)", vectors_key, re.I):
        return f"openai/{vectors_key}"
    return vectors_key


def _balanced_train_pairs(
    samples: List[Tuple[str, Any, int]],
    model_slugs: List[str],
) -> List[Tuple[str, str, Any, int]]:
    """One model per sample: shuffle samples, assign models in round-robin (counts differ by at most 1)."""
    if not model_slugs:
        raise ValueError("model_slugs must be non-empty")
    shuffled = list(samples)
    random.shuffle(shuffled)
    m = len(model_slugs)
    return [(model_slugs[i % m], ds, rec, gi) for i, (ds, rec, gi) in enumerate(shuffled)]


def _product_train_pairs(
    samples: List[Tuple[str, Any, int]],
    model_slugs: List[str],
) -> List[Tuple[str, str, Any, int]]:
    """Legacy: every model for every sample."""
    return [(mod, ds, rec, gi) for mod in model_slugs for ds, rec, gi in samples]


def _expected_spatial_edges(
    logits_flat: torch.Tensor, n: int, temperature: float, topology: str
) -> torch.Tensor:
    fs = flatten_mask(spatial_mask_matrix(topology, n)).to(device=logits_flat.device, dtype=logits_flat.dtype)
    p = torch.sigmoid(logits_flat / temperature)
    return (p * fs).sum()


def _save_checkpoint(
    path: Path,
    head: GraphBasedMASGenerationHead,
    train_args: argparse.Namespace,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train_progress: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    init_args = {
        "role_dim": int(head.role_dim),
        "rel_dim": int(head.rel_dim),
        "hidden_channels": head.gcn1.out_channels,
        "embed_dim": head.gcn2.out_channels,
        "mlp_hidden": head.mlp[0].out_features,
        "dropout": float(head.dropout),
        "gate_alpha": float(head.gate_alpha.item()),
    }
    payload: Dict[str, Any] = {
        "checkpoint_kind": "rcm_card_edge_head",
        "head_state_dict": head.state_dict(),
        "head_init_args": init_args,
        "train_args": vars(train_args),
        "role_activation_checkpoint": str(train_args.role_ckpt),
        "pools_json": str(train_args.pools_json),
        "role_union_embeddings": str(train_args.role_union_embeddings),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if train_progress is not None:
        # Copy so caller can mutate dict after save; snapshot list is already a copy when passed.
        payload["train_progress"] = dict(train_progress)
    torch.save(payload, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train RCM GNN edge head: default full no-anchor data + balanced one-model-per-sample schedule."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        metavar="DS",
        help=f"Dataset slugs when not using --manifest. Default: all supported {SUPPORTED_DATASETS}.",
    )
    p.add_argument("--split", default=SPLIT_TRAIN, choices=[SPLIT_TRAIN, "test"])
    p.add_argument("--limit", type=int, default=0, help="Max samples per dataset (0 = all). Ignored if --manifest is set.")
    p.add_argument(
        "--include-anchor-samples",
        action="store_true",
        help="When loading via --datasets (not --manifest), include anchor-subset rows (default: exclude).",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="",
        help="JSON from scripts/build_gnn_toy_train_manifest.py (train, no-anchor). Overrides --datasets and --limit.",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-cost", type=float, default=0.01, dest="lambda_cost")
    p.add_argument(
        "--no-training-y-json",
        action="store_true",
        help="Do not load offline y; cost uses w=1 (legacy lambda_cost * E_edges only).",
    )
    p.add_argument(
        "--training-y-json",
        action="append",
        default=[],
        metavar="PATH",
        help="sampled_training_data-style JSON (samples[].model/dataset/sample_id/accuracy). "
        "Repeat to merge; later overrides. Default: load training_data_collection/training_data_raw/"
        "sampled_training_data.json if it exists (unless --no-training-y-json).",
    )
    p.add_argument(
        "--y-epsilon",
        type=float,
        default=0.05,
        dest="y_epsilon",
        help="Floor on w(y)=max(y, y_epsilon) when y lookup is loaded.",
    )
    p.add_argument("--gnn-temperature", type=float, default=1.0)
    p.add_argument("--graph-ablation", default="full", choices=["full", "no_gate", "no_hrel", "task_role_only"])
    p.add_argument(
        "--topology",
        default="full",
        choices=["chain", "full", "star", "mesh", "debate"],
        help="Spatial mask for GNN training (candidate directed edges). Default full ≈ legacy rcm_gnn.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Single run: folder under training_data_collection/vectors/ (and OpenRouter slug mapping). "
        "Ignored when --models is non-empty.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Vector folder keys under --vectors-root (and OpenRouter slug mapping when not --mock-llm). "
        "Schedule is controlled by --model-assignment.",
    )
    p.add_argument(
        "--model-assignment",
        default="balanced",
        choices=["balanced", "product"],
        help="balanced: one model per sample, ~equal model counts (shuffle + round-robin each epoch if "
        "--reshuffle-model-assignment-each-epoch). product: every model × every sample (legacy).",
    )
    p.add_argument(
        "--no-reshuffle-model-assignment-each-epoch",
        action="store_true",
        help="For balanced assignment only: fix sample→model mapping after the first shuffle+round-robin; "
        "each epoch only shuffles step order. Default is to rebuild mapping every epoch.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mock-llm", action="store_true", help="Use stub LLM (no API keys).")
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt) from this script: head + Adam + train_progress (epoch/step + optional pair snapshot).",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="If set and --resume is empty and --output exists with train_progress, load from --output and continue.",
    )
    p.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=1,
        metavar="N",
        help="Write --output every N completed optimizer steps (default 1 = each step). "
        "Last step of each epoch is always saved. Larger N reduces disk I/O but coarser crash recovery.",
    )
    p.add_argument("--output", type=str, default=str(_DEFAULT_OUT))
    p.add_argument("--role-ckpt", type=str, default=str(_ROLE_CKPT))
    p.add_argument("--role-profiles", type=str, default=str(_ROLE_PROFILES))
    p.add_argument("--role-union-embeddings", type=str, default=str(_ROLE_UNION_EMB))
    p.add_argument("--pools-json", type=str, default=str(_ROLE_POOLS_JSON))
    p.add_argument("--vectors-root", type=str, default=str(_VECTORS_ROOT))
    args = p.parse_args()

    if int(getattr(args, "checkpoint_every_steps", 1)) < 1:
        raise SystemExit("--checkpoint-every-steps must be >= 1")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.output).expanduser().resolve()
    resume_path: Optional[Path] = None
    if str(args.resume).strip():
        resume_path = Path(str(args.resume).strip()).expanduser().resolve()
    elif args.auto_resume and out_path.is_file():
        resume_path = out_path
        print(f"Auto-resume: loading checkpoint from {out_path}", flush=True)

    model_slugs = [str(x).strip() for x in (args.models or []) if str(x).strip()]
    if not model_slugs:
        model_slugs = [str(args.model).strip()]
    if not model_slugs[0]:
        raise SystemExit("Provide --model or non-empty --models.")

    llm: Any
    if args.mock_llm:
        llm = _MockLLM()
    else:
        from graph_mas.gpt_chat import GPTChat  # noqa: PLC0415

        llm = GPTChat(model_name=_openrouter_slug(model_slugs[0]))

    meta_loaded: Optional[Dict[str, Any]] = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"Resume path is not a file: {resume_path}")
        head, meta_loaded = load_rcm_card_edge_head_with_meta(resume_path, device)
        print(f"Loaded checkpoint from {resume_path}", flush=True)
    else:
        head = GraphBasedMASGenerationHead().to(device)

    runner = MASRunner(
        args.role_ckpt,
        args.role_profiles,
        llm,
        device=str(device),
        role_union_embeddings=args.role_union_embeddings,
        pools_json=args.pools_json,
        rcm_card_graph_head=head,
    )
    runner.pipeline.eval()
    for p_ in runner.pipeline.parameters():
        p_.requires_grad = False

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    if meta_loaded and meta_loaded.get("optimizer_state_dict"):
        try:
            opt.load_state_dict(meta_loaded["optimizer_state_dict"])
            print("Restored Adam optimizer state from checkpoint.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not load optimizer state ({exc}); using fresh Adam.", flush=True)

    prog0 = (meta_loaded or {}).get("train_progress") or {}
    start_epoch = int(prog0.get("epoch_next", 0))
    start_step_in_epoch = int(prog0.get("next_step_in_epoch", 0))
    resume_pairs_snapshot: Any = prog0.get("epoch_train_pairs_snapshot")
    global_step = int(prog0.get("global_step", 0))
    missing_y = int(prog0.get("missing_y", 0))
    if meta_loaded and prog0:
        snap_note = "with epoch_train_pairs_snapshot" if resume_pairs_snapshot else "no pair snapshot"
        print(
            f"Resume progress: epochs [ {start_epoch}, {args.epochs} ), "
            f"next_step_in_epoch={start_step_in_epoch} ({snap_note}), "
            f"global_step={global_step}, missing_y={missing_y}",
            flush=True,
        )
    elif meta_loaded and not prog0:
        print(
            "Note: checkpoint has no train_progress; running from epoch 0 with loaded weights "
            "(Adam state restored only if optimizer_state_dict was present).",
            flush=True,
        )

    if args.manifest:
        samples = _load_samples_from_manifest(Path(args.manifest).expanduser().resolve(), args.split)
    else:
        samples = []
        exclude_anchors = not bool(args.include_anchor_samples)
        for ds in args.datasets:
            ds = str(ds).lower()
            records, gix, _ = _build_records_for_eval(ds, args.split, exclude_anchor_samples=exclude_anchors)
            if args.limit and args.limit > 0:
                records = records[: args.limit]
                gix = gix[: args.limit]
            for rec, gi in zip(records, gix):
                samples.append((ds, rec, int(gi)))

    y_paths_from_user = False
    if args.no_training_y_json:
        y_paths: List[str] = []
    else:
        user_paths = [str(x) for x in (args.training_y_json or []) if str(x).strip()]
        if user_paths:
            y_paths = user_paths
            y_paths_from_user = True
        else:
            dp = _DEFAULT_TRAINING_Y_JSON
            y_paths = [str(dp)] if dp.is_file() else []
            if not y_paths:
                print(
                    f"Note: default y-json not found ({dp}); cost uses w=1. "
                    f"Place sampled_training_data.json there or pass --training-y-json.",
                    flush=True,
                )
    y_lookup: Optional[Dict[Tuple[str, str, str], float]] = _load_y_accuracy_lookup(y_paths) if y_paths else None
    if y_lookup:
        print(f"Loaded y=accuracy lookup: {len(y_lookup)} keys from {len(y_paths)} JSON file(s).")
    y_eps = float(args.y_epsilon)
    if y_eps < 0.0 or y_eps > 1.0:
        raise SystemExit("--y-epsilon should be in [0, 1].")

    if samples and len(samples[0]) != 3:
        raise RuntimeError("internal: samples entries must be (ds, rec, global_index)")

    n_models = len(model_slugs)
    n_tasks = len(samples)
    balanced = str(args.model_assignment).lower() == "balanced"
    reshuffle_assignment_each_epoch = balanced and (not args.no_reshuffle_model_assignment_each_epoch)

    if balanced:
        base_schedule_pairs = _balanced_train_pairs(samples, model_slugs)
    else:
        base_schedule_pairs = _product_train_pairs(samples, model_slugs)
    steps_per_epoch = len(base_schedule_pairs)

    if not base_schedule_pairs:
        raise SystemExit("No training pairs (empty samples or empty --models).")

    total_steps = steps_per_epoch * args.epochs
    if y_lookup:
        cost_suffix = (
            f"; cost w(y)=max(y,{y_eps}) from --training-y-json"
            if y_paths_from_user
            else f"; cost w(y)=max(y,{y_eps}) from default {_DEFAULT_TRAINING_Y_JSON.name}"
        )
    else:
        cost_suffix = ""
    if balanced:
        c0 = Counter(m for m, _, _, _ in base_schedule_pairs)
        lo, hi = (n_tasks // n_models) if n_models else 0, ((n_tasks + n_models - 1) // n_models) if n_models else 0
        resh = "reshuffle mapping each epoch" if reshuffle_assignment_each_epoch else "fixed mapping"
        print(
            f"Training schedule (balanced): {n_tasks} task(s), {n_models} model(s) → {steps_per_epoch} steps/epoch "
            f"({resh}; per-model counts in [{lo},{hi}] per epoch; example draw: {dict(c0)}). models={model_slugs!r}"
            + cost_suffix
        )
    else:
        print(
            f"Training schedule (product): {n_models} model(s) × {n_tasks} task(s) = {steps_per_epoch} steps/epoch "
            f"(shuffled each epoch). models={model_slugs!r}"
            + cost_suffix
        )

    if start_epoch >= args.epochs:
        print(
            f"Nothing to train: resume epoch_next={start_epoch} >= --epochs={args.epochs}. "
            f"Checkpoint: {out_path}",
            flush=True,
        )
        return

    ckpt_every = int(args.checkpoint_every_steps)

    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and start_step_in_epoch > 0 and isinstance(resume_pairs_snapshot, list):
            train_pairs = list(resume_pairs_snapshot)
        elif epoch == start_epoch and start_step_in_epoch > 0:
            print(
                "Warning: checkpoint has next_step_in_epoch>0 but no epoch_train_pairs_snapshot; "
                "rebuilding schedule and restarting this epoch from step 0.",
                flush=True,
            )
            if balanced:
                if reshuffle_assignment_each_epoch:
                    train_pairs = _balanced_train_pairs(samples, model_slugs)
                else:
                    train_pairs = list(base_schedule_pairs)
                    random.shuffle(train_pairs)
            else:
                train_pairs = list(base_schedule_pairs)
                random.shuffle(train_pairs)
            start_step_in_epoch = 0
        elif balanced:
            if reshuffle_assignment_each_epoch:
                train_pairs = _balanced_train_pairs(samples, model_slugs)
            else:
                train_pairs = list(base_schedule_pairs)
                random.shuffle(train_pairs)
        else:
            train_pairs = list(base_schedule_pairs)
            random.shuffle(train_pairs)

        if start_step_in_epoch > len(train_pairs):
            print(
                f"Warning: next_step_in_epoch={start_step_in_epoch} > len(train_pairs)={len(train_pairs)}; clamping to 0.",
                flush=True,
            )
            start_step_in_epoch = 0

        for j, (vec_key, ds, rec, gi) in enumerate(train_pairs):
            if epoch == start_epoch and j < start_step_in_epoch:
                continue
            i_in_epoch = j + 1
            vm = MASRunner.load_vm_for_model(vec_key, args.vectors_root, device=str(device))
            if not args.mock_llm and hasattr(runner.llm, "model_name"):
                runner.llm.model_name = _openrouter_slug(vec_key)

            cfg = get_evaluation_config(ds)
            inp = record_to_input(rec, ds)
            inp["dataset"] = ds

            head.train()
            graph, ctx, logits, meta = runner.build_rcm_gnn_graph(
                inp,
                vm,
                dataset=ds,
                topology=args.topology,
                graph_ablation=args.graph_ablation,
            )

            g_seed = random.randint(0, 2**31 - 1)
            last_out, log_prob = graph.run(
                inp,
                num_rounds=1,
                seed=g_seed,
                temperature=args.gnn_temperature,
                threshold=None,
            )

            raw = runner._decide(
                meta["task_text"],
                last_out,
                ctx.selected_roles,
                decision_format_constraint=meta["decision_format_constraint"],
            )
            post_with_record = cfg.get("postprocess_answer_with_record")
            post = (
                post_with_record(raw, rec)
                if post_with_record is not None
                else cfg["postprocess_answer"](raw)
            )
            ok = bool(cfg["check_correctness"](post, rec))
            utility = torch.tensor(1.0 if ok else 0.0, device=device, dtype=log_prob.dtype)

            n_agents = len(ctx.selected_roles)
            expected_edges = _expected_spatial_edges(
                logits, n_agents, args.gnn_temperature, args.topology
            )
            if y_lookup is not None:
                y_val = _lookup_training_y(y_lookup, vec_key, ds, gi)
                if y_val is None:
                    missing_y += 1
                    y_val = 1.0
                w_cost = max(float(y_val), y_eps)
            else:
                y_val = None
                w_cost = 1.0
            loss = -log_prob * utility + args.lambda_cost * w_cost * expected_edges

            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1
            y_disp = f"{float(y_val):.2f}" if y_val is not None else "-"
            print(
                f"[{i_in_epoch}/{steps_per_epoch}] ep={epoch + 1}/{args.epochs} "
                f"total={global_step}/{total_steps} vm={vec_key} ds={ds} "
                f"loss={float(loss.detach()):.4f} utility={float(utility):.0f} "
                f"logp={float(log_prob.detach()):.3f} E_edges={float(expected_edges.detach()):.2f} "
                f"w_cost={w_cost:.3f} y={y_disp} agents={n_agents} topo={args.topology}",
                flush=True,
            )

            next_s = j + 1
            if next_s < len(train_pairs):
                prog_save: Dict[str, Any] = {
                    "epoch_next": epoch,
                    "next_step_in_epoch": next_s,
                    "global_step": global_step,
                    "missing_y": missing_y,
                    "epoch_train_pairs_snapshot": list(train_pairs),
                }
            else:
                prog_save = {
                    "epoch_next": epoch + 1,
                    "next_step_in_epoch": 0,
                    "global_step": global_step,
                    "missing_y": missing_y,
                }
            if (global_step % ckpt_every == 0) or (j == len(train_pairs) - 1):
                _save_checkpoint(out_path, head, args, optimizer=opt, train_progress=prog_save)

    if y_lookup is not None and missing_y > 0:
        print(
            f"Note: {missing_y} step(s) had no (model,dataset,sample_id) match in --training-y-json; "
            f"used y=1.0 for w_cost floor branch."
        )


if __name__ == "__main__":
    main()
