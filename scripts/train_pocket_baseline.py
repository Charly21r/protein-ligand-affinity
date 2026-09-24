"""Pocket-only baseline.

Strips the ligand entirely and predicts affinity from the binding site alone.

    .venv/bin/python scripts/train_pocket_baseline.py

This is the companion to train_ligand_baseline.py and the other half of Volkov
et al.'s argument. A pocket has no information about *which drug* is bound, so
whatever this scores is a measure of how much affinity is predictable from
protein identity - the shortcut the Phase 4 adversarial component targets.

Two feature sets are reported because ATOM3D's pockets are defined by proximity
to the ligand, so pocket size is a proxy for ligand size (r = +0.63). The
size-free ``composition`` set is the honest number; ``composition_size`` shows
how much is bought by the confound. See src/lba/models/pocket_only.py.

Writes results/tables/pocket_baseline_id{identity}.csv.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from lba.data.lba import SPLITS  # noqa: E402
from lba.eval.metrics import format_metrics, regression_metrics  # noqa: E402
from lba.models.pocket_only import (  # noqa: E402
    FEATURE_SETS,
    feature_matrix,
    pocket_table,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "tables"

# Same convention as the ligand baseline: RidgeCV is deterministic given the
# data, so looping it over seeds would manufacture a standard deviation of zero.
STOCHASTIC = {"ridge": False, "rf": True}


def build_models(seed: int) -> dict:
    """Model name -> estimator. Identical to the ligand baseline on purpose."""
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "rf": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=int, default=30, choices=[30, 60])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    print(f"Loading LBA pockets (sequence-identity-{args.identity}) ...")
    print("  (first run walks the LMDB and takes a few minutes; then cached)")
    tables = {}
    for split in SPLITS:
        tables[split] = pocket_table(split, identity=args.identity)
        frame = tables[split]
        print(
            f"  {split:<6} {len(frame):>5} pockets  "
            f"median {frame.n_residues.median():.0f} residues, "
            f"{frame.n_heavy_atoms.median():.0f} heavy atoms"
        )

    rows = []

    # --- Floor: predict the training mean. ---
    x_train = feature_matrix(tables["train"], "composition")
    dummy = DummyRegressor(strategy="mean").fit(x_train, tables["train"]["label"])
    for split in ("val", "test"):
        scores = regression_metrics(
            tables[split]["label"],
            dummy.predict(feature_matrix(tables[split], "composition")),
        )
        rows.append({"model": "predict_mean", "features": "-", "seed": 0, "split": split, **scores})

    # --- Learned pocket-only models. ---
    for feature_set in FEATURE_SETS:
        features = {s: feature_matrix(tables[s], feature_set) for s in SPLITS}
        for model_name in build_models(args.seeds[0]):
            seeds = args.seeds if STOCHASTIC[model_name] else args.seeds[:1]
            for seed in seeds:
                model = build_models(seed)[model_name]
                model.fit(features["train"], tables["train"]["label"])
                for split in ("val", "test"):
                    scores = regression_metrics(
                        tables[split]["label"], model.predict(features[split])
                    )
                    rows.append({
                        "model": model_name, "features": feature_set,
                        "seed": seed, "split": split, **scores,
                    })

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pocket_baseline_id{args.identity}.csv"
    frame.to_csv(path, index=False)

    print(f"\n{'=' * 72}\nTEST SPLIT (mean over seeds)\n{'=' * 72}")
    test = frame[frame["split"] == "test"]
    grouped = test.groupby(["model", "features"])
    summary = (
        grouped[["pearson_r", "spearman_r", "rmse", "mae"]]
        .mean()
        .sort_values("pearson_r", ascending=False)
    )
    spread = grouped["pearson_r"].agg(["std", "count"])
    for (model_name, feature_set), row in summary.iterrows():
        std, count = spread.loc[(model_name, feature_set)]
        tail = f"  (n={int(count)})" if count == 1 else f"  (r sd={std:.3f}, n={int(count)})"
        print(format_metrics(f"{model_name} [{feature_set}]", row.to_dict()) + tail)

    # The comparison the whole script exists for.
    best = {
        fs: summary.xs(fs, level="features")["pearson_r"].max() for fs in FEATURE_SETS
    }
    print(
        f"\nSize confound: best composition-only r = {best['composition']:.3f}, "
        f"best with size features r = {best['composition_size']:.3f} "
        f"(delta {best['composition_size'] - best['composition']:+.3f}).\n"
        "Pocket size correlates r = +0.63 with ligand size because ATOM3D defines\n"
        "the pocket by proximity to the ligand, so that delta is an upper bound on\n"
        "how much 'pocket' signal is really the ligand measured indirectly."
    )
    print(f"\nWrote {path.relative_to(path.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
