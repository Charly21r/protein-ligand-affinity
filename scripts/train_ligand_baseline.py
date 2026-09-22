"""Ligand-only baseline.

Strips the protein entirely and predicts affinity from the ligand alone.

    .venv/bin/python scripts/train_ligand_baseline.py

Under the project hypothesis a high score here is the *finding*, not a success:
it means PDBbind affinities are largely predictable from ligand identity, so a
full interaction model that fails to beat this is not using the interaction.

Writes results/tables/ligand_baseline_id{identity}.csv.
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

from lba.data.lba import load_all_records  # noqa: E402
from lba.eval.metrics import format_metrics, regression_metrics  # noqa: E402
from lba.models.ligand_only import (  # noqa: E402
    ALL_DESCRIPTORS,
    DESCRIPTORS,
    descriptor_features,
    morgan_fingerprints,
    nearest_neighbour_predict,
    parse_smiles,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "tables"

# Which models actually consume their seed. RidgeCV is deterministic given the data
STOCHASTIC = {"ridge": False, "rf": True}


def build_models(seed: int) -> dict:
    """Model name -> estimator. Kept small and standard on purpose."""
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
    parser.add_argument(
        "--strict-smiles",
        action="store_true",
        help="Drop SMILES that fail RDKit's valence check instead of recovering "
        "them. Those ligands skew ~0.8 log units weaker, so this biases the "
        "training set - use only to measure that effect.",
    )
    args = parser.parse_args()

    print(f"Loading LBA (sequence-identity-{args.identity}) ...")
    records = load_all_records(identity=args.identity)

    data = {}
    for split, split_records in records.items():
        smiles = [r.smiles for r in split_records]
        labels = np.array([r.label for r in split_records], dtype=float)
        mols, valid, n_relaxed = parse_smiles(smiles, allow_relaxed=not args.strict_smiles)
        dropped = int((~valid).sum())
        data[split] = {
            "mols": mols,
            "y": labels[valid],
            "smiles": [s for s, ok in zip(smiles, valid) if ok],
        }
        note = f" ({n_relaxed} needed relaxed valence parsing)" if n_relaxed else ""
        print(f"  {split:<6} {len(mols):>5} ligands{note}")
        if dropped:
            print(f"  [{split}] {dropped} SMILES unparseable even relaxed, dropped")

    # Descriptor medians are fitted on train only and reused for val/test, so no
    # evaluation-split statistic reaches its own features.
    features = {split: {} for split in data}
    for name, descriptor_names in (
        ("descriptors", DESCRIPTORS),
        ("descriptors_full", ALL_DESCRIPTORS),
    ):
        features["train"][name], medians = descriptor_features(
            data["train"]["mols"], descriptor_names
        )
        for split in ("val", "test"):
            features[split][name], _ = descriptor_features(
                data[split]["mols"], descriptor_names, medians=medians
            )
    for split, payload in data.items():
        features[split]["morgan"] = morgan_fingerprints(payload["mols"])

    print(
        "\nfeature dims: "
        + ", ".join(
            f"{name}={features['train'][name].shape[1]}"
            for name in ("descriptors", "descriptors_full", "morgan")
        )
    )

    rows = []

    # --- Floor: predict the training mean. Any model must beat this. ---
    dummy = DummyRegressor(strategy="mean").fit(
        features["train"]["descriptors"], data["train"]["y"]
    )
    for split in ("val", "test"):
        scores = regression_metrics(
            data[split]["y"], dummy.predict(features[split]["descriptors"])
        )
        rows.append({"model": "predict_mean", "features": "-", "seed": 0, "split": split, **scores})

    # --- Retrieval probe: no learning, just copy the nearest training ligand. ---
    # The 30% split separates proteins, not ligands, so ligand leakage survives
    # it. frac_exact_duplicate is the sharpest measure of that: the same molecule
    # verbatim, against an unrelated protein.
    print("\nRunning Tanimoto nearest-neighbour probe ...")
    train_smiles = set(data["train"]["smiles"])
    for split in ("val", "test"):
        predictions, similarity = nearest_neighbour_predict(
            data["train"]["mols"], data["train"]["y"], data[split]["mols"]
        )
        exact = float(
            np.mean([s in train_smiles for s in data[split]["smiles"]])
        )
        scores = regression_metrics(data[split]["y"], predictions)
        rows.append({
            "model": "nn_tanimoto_k1", "features": "morgan", "seed": 0,
            "split": split, **scores,
            "mean_max_similarity": float(similarity.mean()),
            "frac_near_duplicate": float((similarity > 0.9).mean()),
            "frac_exact_duplicate": exact,
        })
        print(
            f"  {split}: mean max Tanimoto to train = {similarity.mean():.3f}, "
            f"{(similarity > 0.9).mean():.1%} of ligands have a >0.9 match, "
            f"{exact:.1%} are exact SMILES duplicates"
        )

    # --- Learned ligand-only models. ---
    for feature_name in ("descriptors", "descriptors_full", "morgan"):
        for model_name in build_models(args.seeds[0]):
            # Deterministic models are fitted once; see STOCHASTIC.
            seeds = args.seeds if STOCHASTIC[model_name] else args.seeds[:1]
            for seed in seeds:
                model = build_models(seed)[model_name]
                model.fit(features["train"][feature_name], data["train"]["y"])
                for split in ("val", "test"):
                    scores = regression_metrics(
                        data[split]["y"], model.predict(features[split][feature_name])
                    )
                    rows.append({
                        "model": model_name, "features": feature_name,
                        "seed": seed, "split": split, **scores,
                    })

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ligand_baseline_id{args.identity}.csv"
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
    for (model_name, feature_name), row in summary.iterrows():
        std, count = spread.loc[(model_name, feature_name)]
        # count == 1 means deterministic (or a single seed): no spread to report.
        tail = f"  (n={int(count)})" if count == 1 else f"  (r sd={std:.3f}, n={int(count)})"
        print(format_metrics(f"{model_name} [{feature_name}]", row.to_dict()) + tail)

    print(f"\nWrote {path.relative_to(path.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
