"""Regenerate every diagnostic number quoted in the lab notebook.

    .venv/bin/python scripts/analyze_baselines.py

The training scripts report how the baselines *scored*. This reports what they
are actually using, and how confident any of it is. Every table it writes backs
a claim that would otherwise exist only as prose.

It refits the baselines rather than reading predictions from the training
scripts, so it is self-contained and cannot silently analyse a stale model. That
costs a few minutes, mostly RDKit descriptors; the pocket table is cached.

Writes results/tables/analysis_*.csv.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from lba.data.lba import load_all_records  # noqa: E402
from lba.eval.metrics import regression_metrics  # noqa: E402
from lba.eval.probes import (  # noqa: E402
    collinear_pairs,
    single_feature_baseline,
    single_feature_scan,
    smiles_quality_report,
    stratify_by_similarity,
    within_ligand_variance,
)
from lba.eval.stats import (  # noqa: E402
    calibration_diagnostics,
    paired_bootstrap_delta,
    pearson_ci,
)
from lba.models.ligand_only import (  # noqa: E402
    ALL_DESCRIPTORS,
    DESCRIPTORS,
    descriptor_features,
    nearest_neighbour_predict,
    parse_smiles,
)
from lba.models.pocket_only import (  # noqa: E402
    FEATURE_SETS,
    feature_matrix,
    pocket_table,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "tables"


def _fit_predict(x_train, y_train, x_test, seeds) -> np.ndarray:
    """Seed-averaged RF predictions, matching the baselines' configuration."""
    return np.mean(
        [
            RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
            )
            .fit(x_train, y_train)
            .predict(x_test)
            for seed in seeds
        ],
        axis=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=int, default=30, choices=[30, 60])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=3000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name: str, frame: pd.DataFrame) -> None:
        path = OUT / f"analysis_{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path.name)

    # ------------------------------------------------------------------ data
    print(f"Loading LBA (sequence-identity-{args.identity}) ...")
    records = load_all_records(identity=args.identity)
    ligand = {}
    for split, rows in records.items():
        smiles = [r.smiles for r in rows]
        labels = np.array([r.label for r in rows], dtype=float)
        mols, valid, _ = parse_smiles(smiles)
        ligand[split] = {
            "mols": mols,
            "y": labels[valid],
            "smiles": [s for s, ok in zip(smiles, valid) if ok],
            "ids": [r.pdb_id for r, ok in zip(rows, valid) if ok],
        }

    # ------------------------------------------- 1. what the SMILES encode
    print("1/7 SMILES quality ...")
    write("smiles_quality", pd.DataFrame([
        {"split": split, **smiles_quality_report(
            [r.smiles for r in records[split]],
            [r.label for r in records[split]],
        )}
        for split in records
    ]))

    # ------------------------------ 2. same ligand, different protein
    print("2/7 within-ligand affinity variance ...")
    every = [(r.smiles, r.label) for rows in records.values() for r in rows]
    repeated, summary = within_ligand_variance(
        [s for s, _ in every], [y for _, y in every]
    )
    write("within_ligand_summary", pd.DataFrame([summary]))
    write("within_ligand_repeated", repeated.head(50))
    print(f"      median within-ligand SD {summary['median_within_ligand_sd']:.2f} pKd, "
          f"max range {summary['max_within_ligand_range']:.2f}, "
          f"oracle RMSE {summary['oracle_rmse']:.3f}")

    # --------------------------------------------- 3. ligand featurization
    print("3/7 ligand descriptors (slow) ...")
    x_train, medians = descriptor_features(ligand["train"]["mols"], ALL_DESCRIPTORS)
    x_test, _ = descriptor_features(ligand["test"]["mols"], ALL_DESCRIPTORS, medians=medians)
    curated_train, curated_medians = descriptor_features(ligand["train"]["mols"], DESCRIPTORS)

    write("feature_scan_ligand", single_feature_scan(
        x_train, ligand["train"]["y"], ALL_DESCRIPTORS))
    write("collinearity_curated_descriptors", collinear_pairs(curated_train, DESCRIPTORS))

    # --------------------------------------------------- 4. pocket features
    print("4/7 pocket features ...")
    pockets = {s: pocket_table(s, identity=args.identity) for s in ("train", "test")}
    all_pocket_columns = FEATURE_SETS["composition_size"]
    write("feature_scan_pocket", single_feature_scan(
        feature_matrix(pockets["train"], "composition_size"),
        pockets["train"]["label"].to_numpy(),
        all_pocket_columns,
    ))

    # The pocket-definition confound: ATOM3D cut pockets around the ligand, so
    # pocket size partly measures ligand size. Quantified rather than asserted.
    ligand_size = np.array([m.GetNumHeavyAtoms() for m in ligand["train"]["mols"]])
    train_ids = pd.Index(pockets["train"]["pdb_id"])
    order = train_ids.get_indexer(ligand["train"]["ids"])
    if (order < 0).any():
        raise RuntimeError("pocket table is missing pdb_ids present in the ligand records")
    write("pocket_size_confound", pd.DataFrame([
        {
            "pocket_feature": column,
            "r_with_ligand_heavy_atoms": regression_metrics(
                ligand_size, pockets["train"][column].to_numpy()[order]
            )["pearson_r"],
            "r_with_pkd": regression_metrics(
                pockets["train"]["label"].to_numpy(), pockets["train"][column].to_numpy()
            )["pearson_r"],
        }
        for column in ("n_residues", "n_heavy_atoms", "radius_of_gyration")
    ]))

    # ----------------------------------------------- 5. model predictions
    print("5/7 refitting baselines ...")
    test_ids = pd.Index(pockets["test"]["pdb_id"])
    test_order = test_ids.get_indexer(ligand["test"]["ids"])
    if (test_order < 0).any():
        raise RuntimeError("pocket/ligand test sets do not align by pdb_id")
    y_test = ligand["test"]["y"]

    # The bar Phase 2 actually has to clear: single features fitted on train and
    # scored on test. A pocket residue count alone beats the whole ligand model.
    pocket_train = feature_matrix(pockets["train"], "composition_size")
    pocket_test = feature_matrix(pockets["test"], "composition_size")
    single = pd.concat([
        single_feature_baseline(
            x_train, ligand["train"]["y"], x_test, y_test, ALL_DESCRIPTORS,
            features=["HeavyAtomCount", "MolWt", "MolMR", "LabuteASA"],
        ).assign(source="ligand"),
        single_feature_baseline(
            pocket_train, pockets["train"]["label"].to_numpy(),
            pocket_test, pockets["test"]["label"].to_numpy(),
            all_pocket_columns,
            features=["n_residues", "n_heavy_atoms", "radius_of_gyration"],
        ).assign(source="pocket"),
    ], ignore_index=True).sort_values("pearson_r", ascending=False)
    write("single_feature_baselines", single)
    best = single.iloc[0]
    print(f"      strongest single feature: {best.feature} ({best.source}) "
          f"test r={best.pearson_r:+.3f} rmse={best.rmse:.3f}")

    predictions = {
        "ligand_descriptors_full": _fit_predict(
            x_train, ligand["train"]["y"], x_test, args.seeds
        )
    }
    for feature_set in FEATURE_SETS:
        columns = [all_pocket_columns.index(c) for c in FEATURE_SETS[feature_set]]
        predictions[f"pocket_{feature_set}"] = _fit_predict(
            pocket_train[:, columns],
            pockets["train"]["label"].to_numpy(),
            pocket_test[test_order][:, columns],
            args.seeds,
        )
    _, similarity = nearest_neighbour_predict(
        ligand["train"]["mols"], ligand["train"]["y"], ligand["test"]["mols"]
    )
    write("predictions", pd.DataFrame(
        {"pdb_id": ligand["test"]["ids"], "y_true": y_test,
         "max_tanimoto_to_train": similarity, **predictions}
    ))

    # ------------------------------ 6. intervals, calibration, stratification
    print("6/7 intervals and calibration ...")
    write("model_intervals", pd.DataFrame([
        {"model": name, **pearson_ci(y_test, pred),
         **calibration_diagnostics(y_test, pred)}
        for name, pred in predictions.items()
    ]))
    stratified = []
    for name, pred in predictions.items():
        frame = stratify_by_similarity(y_test, pred, similarity)
        frame.insert(0, "model", name)
        stratified.append(frame)
    write("stratified_by_similarity", pd.concat(stratified, ignore_index=True))

    # --------------------------------------------- 7. paired comparisons
    print("7/7 paired bootstrap ...")
    reference = "ligand_descriptors_full"
    comparisons = []
    for name, pred in predictions.items():
        if name == reference:
            continue
        for metric in ("pearson_r", "rmse"):
            comparisons.append({
                "reference": reference, "model": name, "metric": metric,
                **paired_bootstrap_delta(
                    y_test, predictions[reference], pred,
                    metric=metric, n_boot=args.n_boot
                ),
            })
    frame = pd.DataFrame(comparisons)
    write("paired_comparisons", frame)
    for row in frame[frame.metric == "pearson_r"].itertuples():
        verdict = "significant" if row.significant else "not resolvable"
        print(f"      {row.model} vs {reference}: delta r = {row.delta:+.3f} "
              f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]  {verdict}")

    print(f"\nWrote {len(written)} tables to {OUT.relative_to(OUT.parents[1])}/:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
