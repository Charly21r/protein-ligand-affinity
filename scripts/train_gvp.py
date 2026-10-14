#!/usr/bin/env python3
"""Train the GVP-GNN interaction-aware model on ATOM3D LBA.

Sanity-check the pipeline first - this must reach near-zero loss in a couple of
minutes on CPU, and if it does not, nothing below is worth running::

    .venv/bin/python scripts/train_gvp.py --overfit 16 --epochs 120

Then the real thing, which wants a GPU::

    .venv/bin/python scripts/train_gvp.py --epochs 60 --seeds 0 1 2

This is the model the two baselines exist to be compared against. Both of them
were deliberately blinded - ligand-only scores r = 0.516, size-free pocket-only
r = 0.477, and a paired bootstrap cannot separate them - so the question here is
whether a model that sees the geometry *between* the two halves does better than
either half alone.

The bar is higher than it looks, and it is not predict-the-mean
-----------------------------------------------------------------
Three numbers from ``notebook.md`` that this model has to be read against:

* **r = 0.576** - a linear fit on ``n_residues``, the pocket residue count. One
  integer, no chemistry, no learning. A GVP scoring 0.58 has learned nothing a
  residue count does not already give.
* **r = 0.606** - pocket+size on the novel-ligand stratum (max Tanimoto to
  train < 0.4). That stratum is where an interaction model has to prove itself,
  because it is where ligand identity stops being available as a shortcut.
* **r = 0.516** - ligand-only overall, the headline comparison.

So this script writes per-complex test predictions as well as summary metrics,
keyed on ``pdb_id``, so that ``src/lba/eval/stats.paired_bootstrap_delta`` can
run the paired comparison against the existing baselines rather than leaving two
unpaired point estimates to be eyeballed.

Writes ``results/tables/gvp_baseline_id{identity}.csv`` (metrics, same schema as
the baselines) and ``results/tables/gvp_predictions_id{identity}.csv``
(per-complex test predictions, joinable onto ``analysis_predictions.csv``).

Checkpointing is per-epoch and resumable, because Colab kills sessions at ~12 h
and sooner when idle. Point ``--checkpoint-dir`` at Drive there, not at
``/content``, or a killed session is a lost run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from lba.data.graph import build_split_graphs, graph_dims  # noqa: E402
from lba.data.lba import DEFAULT_CACHE, DEFAULT_RAW, SPLITS  # noqa: E402
from lba.eval.metrics import format_metrics, regression_metrics  # noqa: E402
from lba.models.gvp import GVPAffinityModel  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "tables"


def make_model(args: argparse.Namespace, n_rbf: int) -> GVPAffinityModel:
    """Build the model with dimensions read from the graph builder.

    Dimensions come from ``graph_dims`` rather than being written out here, so
    that changing the element vocabulary or the number of RBFs cannot silently
    desynchronise the data from the model.
    """
    dims = graph_dims(n_rbf)
    return GVPAffinityModel(
        node_in_dims=(dims["node_s"], dims["node_v"]),
        edge_in_dims=(dims["edge_s"], dims["edge_v"]),
        node_h_dims=(args.hidden_scalar, args.hidden_vector),
        n_layers=args.n_layers,
        dropout=args.dropout,
    )


@torch.no_grad()
def evaluate(
    model: GVPAffinityModel, loader, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    """Run the model over a loader and score it.

    Returns the metrics plus the raw predictions, truths and ids, because the
    per-complex predictions are needed for the paired bootstrap and recomputing
    them later would mean a second forward pass over the split.
    """
    model.eval()
    predictions, truths, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).float().cpu().numpy())
        truths.append(batch.y.float().cpu().numpy())
        ids.extend(batch.pdb_id)
    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)
    return regression_metrics(truths, predictions), predictions, truths, ids


def train_one_seed(
    args: argparse.Namespace,
    graphs: dict[str, list],
    seed: int,
    device: torch.device,
    n_rbf: int,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Train one seed to convergence and evaluate on val and test.

    Model selection is on **validation Pearson r**, not on validation loss. They
    disagree: MSE is dominated by the few complexes at the extremes of the pKd
    range, while r is what every number in this project is reported as, and
    selecting on one while reporting the other is a quiet way to look better
    than you are.
    """
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = {
        split: DataLoader(
            graphs[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split in graphs
    }

    model = make_model(args, n_rbf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Plateau scheduling on the selection metric, so the schedule and the
    # selection criterion cannot disagree about what "better" means.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )
    loss_fn = nn.MSELoss()

    checkpoint_dir = args.checkpoint_dir / f"seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    start_epoch, best_r, best_epoch, history = 0, -np.inf, -1, []
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_r, best_epoch, history = state["best_r"], state["best_epoch"], state["history"]
        print(f"  resumed seed {seed} from epoch {start_epoch} (best r {best_r:+.3f})")
        # A seed that already early-stopped must not train further on resume.
        # Without this, re-running all seeds after a killed session gives every
        # finished seed one extra epoch - and if that epoch happens to set a new
        # best val r, it silently changes which model gets selected.
        if args.early_stop and state["epoch"] - best_epoch >= args.early_stop:
            print(f"  seed {seed} had already early-stopped at epoch {state['epoch']}, skipping training")
            start_epoch = args.epochs

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, total_loss, n_batches = time.time(), 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y.view(-1).to(model_dtype(model)))
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        if not np.isfinite(train_loss):
            # Node vectors start at exactly zero, where the gradient of a plain
            # L2 norm is infinite; lba.models.gvp._norm guards that. A NaN here
            # means the guard failed or the LR is far too high, and continuing
            # would waste the whole run.
            print(f"  epoch {epoch}: non-finite loss ({train_loss}), aborting seed")
            break

        val_scores, *_ = evaluate(model, loaders["val"], device)
        scheduler.step(val_scores["pearson_r"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val_scores})

        marker = ""
        if val_scores["pearson_r"] > best_r:
            best_r, best_epoch, marker = val_scores["pearson_r"], epoch, "  *"
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_r": best_r,
                "best_epoch": best_epoch,
                "history": history,
            },
            last_path,
        )

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:>4}  loss {train_loss:.4f}  "
                f"val r {val_scores['pearson_r']:+.3f}  val rmse {val_scores['rmse']:.3f}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                f"{time.time() - started:.0f}s{marker}"
            )

        if args.early_stop and epoch - best_epoch >= args.early_stop:
            print(f"  no val improvement in {args.early_stop} epochs, stopping")
            break

    # Restore the selected epoch before scoring test. Scoring the *final* epoch
    # instead is a common and consequential slip: it reports a model nobody
    # selected.
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model"]
        )
        print(f"  restored epoch {best_epoch} (val r {best_r:+.3f})")

    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))

    scores: dict[str, dict[str, float]] = {}
    predictions_frame = None
    for split in ("val", "test"):
        scores[split], predictions, truths, ids = evaluate(model, loaders[split], device)
        if split == "test":
            predictions_frame = pd.DataFrame(
                {"pdb_id": ids, "y_true": truths, f"gvp_seed{seed}": predictions}
            )
    return scores, predictions_frame


def model_dtype(model: nn.Module) -> torch.dtype:
    """The dtype the model's parameters are in, for casting labels to match."""
    return next(model.parameters()).dtype


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=int, default=30, choices=[30, 60])
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "checkpoints" / "gvp",
        help="on Colab, point this at Drive - /content is wiped when the session dies",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-scalar", type=int, default=100)
    parser.add_argument("--hidden-vector", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience on val r",
    )
    parser.add_argument("--early-stop", type=int, default=15, help="0 disables")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1, help="epochs between log lines")
    parser.add_argument("--pocket-cutoff", type=float, default=8.0)
    parser.add_argument("--edge-cutoff", type=float, default=5.0)
    parser.add_argument("--n-rbf", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    parser.add_argument("--rebuild-graphs", action="store_true")
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help=(
            "sanity mode: train on this many complexes and use them as val and "
            "test too. Loss must fall to near zero. Nothing is written."
        ),
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"  {properties.name}, {properties.total_memory / 1e9:.1f} GB")

    print(f"\nBuilding graphs (identity {args.identity}) ...")
    print("  (first run walks the LMDB and takes a few minutes; then cached)")
    graphs = {}
    for split in SPLITS:
        graphs[split] = build_split_graphs(
            split,
            root=args.root,
            identity=args.identity,
            cache_dir=args.cache_dir,
            pocket_cutoff=args.pocket_cutoff,
            edge_cutoff=args.edge_cutoff,
            n_rbf=args.n_rbf,
            rebuild=args.rebuild_graphs,
        )
        nodes = [g.num_nodes for g in graphs[split]]
        edges = [g.edge_index.shape[1] for g in graphs[split]]
        print(
            f"  {split:<6} {len(graphs[split]):>5} graphs  "
            f"median {int(np.median(nodes))} nodes / {int(np.median(edges))} edges  "
            f"max {max(nodes)} nodes"
        )

    if args.overfit:
        # Deliberately train, select and score on the same handful of complexes.
        # This is not a result, it is a test that gradients flow, batching is
        # right and the cache path works. A model with 571k parameters must be
        # able to memorise 16 examples; if it cannot, something is broken.
        subset = graphs["train"][: args.overfit]
        graphs = {"train": subset, "val": subset, "test": subset}
        args.early_stop = 0
        args.seeds = args.seeds[:1]
        args.dropout = 0.0
        args.log_every = max(args.log_every, 50)
        print(
            f"\nOVERFIT SANITY MODE on {len(subset)} complexes: dropout off, "
            f"{args.epochs} epochs.\n"
            "Loss must approach zero and r approach 1.0. For reference, "
            "predicting the mean of this subset gives "
            f"MSE {float(np.var([float(g.y) for g in subset])):.2f} - a 'loss' "
            "near that value is a model that has learned nothing."
        )

    all_rows, prediction_frames = [], []
    for seed in args.seeds:
        print(f"\n{'=' * 72}\nseed {seed}\n{'=' * 72}")
        scores, predictions = train_one_seed(args, graphs, seed, device, args.n_rbf)
        for split, split_scores in scores.items():
            all_rows.append(
                {
                    "model": "gvp",
                    "features": "complex",
                    "seed": seed,
                    "split": split,
                    **split_scores,
                }
            )
            print(format_metrics(f"seed {seed} [{split}]", split_scores))
        if predictions is not None:
            prediction_frames.append(predictions)

    if args.overfit:
        final = all_rows[-1]
        verdict = "PASS" if final["pearson_r"] > 0.95 else "FAIL"
        print(
            f"\n{verdict}: overfit r = {final['pearson_r']:+.3f}, "
            f"rmse = {final['rmse']:.3f}\n"
            "Nothing written - this was a pipeline test, not a result."
        )
        return 0 if verdict == "PASS" else 1

    frame = pd.DataFrame(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / f"gvp_baseline_id{args.identity}.csv"
    frame.to_csv(metrics_path, index=False)

    # Per-complex predictions, one column per seed plus the seed-mean. Joinable
    # onto analysis_predictions.csv by pdb_id for the paired bootstrap. Note the
    # notebook's standing caveat: the mean of per-seed r and the r of the mean
    # prediction are different quantities. Both are available here; the training
    # scripts' CSVs stay canonical for headline scores.
    merged = prediction_frames[0]
    for extra in prediction_frames[1:]:
        merged = merged.merge(extra.drop(columns=["y_true"]), on="pdb_id", how="inner")
    seed_columns = [c for c in merged.columns if c.startswith("gvp_seed")]
    merged["gvp_complex"] = merged[seed_columns].mean(axis=1)
    predictions_path = OUT / f"gvp_predictions_id{args.identity}.csv"
    merged.to_csv(predictions_path, index=False)

    print(f"\n{'=' * 72}\nTEST SPLIT (mean over {len(args.seeds)} seeds)\n{'=' * 72}")
    test = frame[frame["split"] == "test"]
    summary = test[["pearson_r", "spearman_r", "rmse", "mae"]].mean()
    print(format_metrics("gvp [complex]", summary.to_dict()))
    print(f"  r sd over seeds: {test['pearson_r'].std():.3f}")

    mean_prediction_r = float(
        np.corrcoef(merged["y_true"], merged["gvp_complex"])[0, 1]
    )
    print(f"  r of the seed-mean prediction: {mean_prediction_r:+.3f}")

    # The comparisons that decide whether this model did anything.
    print("\nAgainst the numbers this has to beat (from notebook.md):")
    for name, value in (
        ("n_residues, one integer", 0.576),
        ("ligand-only, overall", 0.516),
        ("pocket+size, novel-ligand stratum", 0.606),
    ):
        delta = summary["pearson_r"] - value
        print(f"  {name:<36} r = {value:.3f}   delta {delta:+.3f}")
    print(
        "\nThese are unpaired point estimates. Run the paired bootstrap before\n"
        "claiming any of these differences are real:\n"
        "  .venv/bin/python scripts/analyze_baselines.py"
    )
    print(f"\nWrote {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {predictions_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
