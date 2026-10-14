#!/usr/bin/env python3
"""Paired comparison of the GVP-GNN against the baselines it has to beat.

    .venv/bin/python scripts/analyze_baselines.py   # must run first
    .venv/bin/python scripts/train_gvp.py           # and this
    .venv/bin/python scripts/analyze_gvp.py

``train_gvp.py`` prints unpaired point estimates, which cannot support a claim.
The 95% CI on ligand-only r = 0.516 is [0.448, 0.578] - about +-0.065 - so a GVP
score anywhere in roughly 0.50-0.62 is indistinguishable from it on a point
estimate alone. What separates them is a *paired* bootstrap: both models scored
on the same resampled complexes, so the shared difficulty of those complexes
cancels and the interval on the difference is far tighter than either model's
own. That is the same comparison the notebook already ran for pocket vs ligand
(delta r = +0.080 [+0.010, +0.150], significant).

Why this is a separate script
-----------------------------
``analyze_baselines.py`` deliberately *refits* every model it analyses, so it
can never report on a stale checkpoint. A neural net cannot be refitted inside
an analysis script, so GVP predictions have to be read from disk instead. Rather
than weaken that guarantee for every baseline, the file-reading path lives here,
with an explicit staleness check: the two prediction tables must agree on
``y_true`` for every shared complex, or the join is rejected.

The reference models
--------------------
Three comparisons matter, in increasing order of how much they hurt:

* **ligand-only** (r = 0.516) - the Volkov replication. A model that never sees
  the protein.
* **n_residues** (r = 0.576) - a linear fit on the pocket residue count. One
  integer, no chemistry, no learning. Its per-complex predictions are not in
  ``analysis_predictions.csv``, which holds only the random-forest baselines, so
  this script refits it from the cached pocket table.
* **pocket+size on the novel-ligand stratum** (r = 0.606) - where an
  interaction model actually has to prove itself, because it is where ligand
  identity stops being available as a shortcut.

Writes ``results/tables/analysis_gvp_intervals.csv``,
``analysis_gvp_paired.csv`` and ``analysis_gvp_stratified.csv``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from lba.eval.probes import stratify_by_similarity  # noqa: E402
from lba.eval.stats import (  # noqa: E402
    calibration_diagnostics,
    paired_bootstrap_delta,
    pearson_ci,
)
from lba.models.pocket_only import pocket_table  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results" / "tables"

# The GVP column written by train_gvp.py: the mean prediction over seeds. The
# notebook's standing caveat applies - the mean of per-seed r and the r of the
# mean prediction are different quantities, and both are reported below.
GVP_COLUMN = "gvp_complex"


def load_tables(identity: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join baseline and GVP test predictions on ``pdb_id``.

    Both tables are per-complex predictions on the same 490-complex test split,
    but they are produced by different scripts from different pipelines, so they
    are joined on ``pdb_id`` and then *checked* rather than assumed aligned. The
    notebook's own paired comparisons note "verified identical pdb_id ordering";
    this does that verification in code.
    """
    baseline_path = OUT / "analysis_predictions.csv"
    gvp_path = OUT / f"gvp_predictions_id{identity}.csv"
    for path, script in (
        (baseline_path, "scripts/analyze_baselines.py"),
        (gvp_path, "scripts/train_gvp.py"),
    ):
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(PROJECT_ROOT)} - run {script} first")

    baselines = pd.read_csv(baseline_path)
    gvp = pd.read_csv(gvp_path)

    baseline_models = [
        c for c in baselines.columns
        if c not in ("pdb_id", "y_true", "max_tanimoto_to_train")
    ]
    seed_columns = sorted(c for c in gvp.columns if c.startswith("gvp_seed"))
    if GVP_COLUMN not in gvp.columns:
        raise SystemExit(f"{gvp_path.name} has no {GVP_COLUMN!r} column")

    merged = baselines.merge(
        gvp[["pdb_id", "y_true", *seed_columns, GVP_COLUMN]],
        on="pdb_id",
        how="inner",
        suffixes=("", "_gvp"),
    )
    dropped = len(baselines) - len(merged)
    if dropped:
        # Expected to be small and non-zero: the ligand pipeline drops complexes
        # with unparseable SMILES, the graph builder drops complexes with no
        # pocket atoms in range. Different filters, so the intersection is
        # smaller than either. Visible rather than silent.
        print(
            f"  note: {dropped} of {len(baselines)} baseline complexes have no GVP "
            f"prediction and were dropped from the comparison"
        )
    if merged.empty:
        raise SystemExit("no complexes in common between the two prediction tables")

    # The staleness guard. If these disagree, one of the tables was generated
    # against a different split, identity cutoff or label source, and every
    # number below would be meaningless.
    disagreement = np.abs(merged["y_true"] - merged["y_true_gvp"])
    if disagreement.max() > 1e-6:
        worst = merged.loc[disagreement.idxmax()]
        raise SystemExit(
            f"y_true disagrees between the tables (worst: {worst['pdb_id']}, "
            f"{worst['y_true']} vs {worst['y_true_gvp']}). One of them is stale - "
            "regenerate both before trusting any comparison."
        )
    merged = merged.drop(columns=["y_true_gvp"])
    return merged, baseline_models, seed_columns


def add_n_residues_baseline(merged: pd.DataFrame, identity: int) -> str:
    """Refit the ``n_residues`` linear baseline and attach its test predictions.

    This is the comparison that carries the report, and it is the one model whose
    per-complex predictions are not already on disk. Fitted on train and scored
    on test, exactly as ``single_feature_baseline`` does it, so the r it
    reproduces should match the 0.576 in ``analysis_single_feature_baselines.csv``
    - and if it does not, something has drifted and the printed value will say so.
    """
    train = pocket_table("train", identity=identity)
    test = pocket_table("test", identity=identity)

    model = LinearRegression().fit(
        train[["n_residues"]].to_numpy(), train["label"].to_numpy()
    )
    predictions = pd.Series(
        model.predict(test[["n_residues"]].to_numpy()), index=pd.Index(test["pdb_id"])
    )

    column = "n_residues_linear"
    merged[column] = merged["pdb_id"].map(predictions).to_numpy()
    if merged[column].isna().any():
        missing = int(merged[column].isna().sum())
        raise SystemExit(f"{missing} test complexes are absent from the pocket table")
    return column


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=int, default=30, choices=[30, 60])
    parser.add_argument("--n-boot", type=int, default=3000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    def write(name: str, frame: pd.DataFrame) -> None:
        path = OUT / f"analysis_gvp_{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path.name)

    print("Loading prediction tables ...")
    merged, baseline_models, seed_columns = load_tables(args.identity)
    print(f"  {len(merged)} test complexes in common, "
          f"{len(baseline_models)} baselines, {len(seed_columns)} GVP seeds")

    print("Refitting the n_residues baseline ...")
    n_residues_column = add_n_residues_baseline(merged, args.identity)

    y_true = merged["y_true"].to_numpy()
    similarity = merged["max_tanimoto_to_train"].to_numpy()
    references = [*baseline_models, n_residues_column]
    all_models = [*references, GVP_COLUMN]

    # ------------------------------------------------- per-seed spread first
    # Before any comparison: how much does the GVP number move with the seed?
    # A delta smaller than the seed spread is not a finding about architecture.
    print("\nGVP per-seed test r:")
    seed_scores = []
    for column in seed_columns:
        scores = pearson_ci(y_true, merged[column].to_numpy())
        seed_scores.append(scores["r"])
        print(f"  {column:<14} r = {scores['r']:+.3f}  [{scores['ci_low']:+.3f}, {scores['ci_high']:+.3f}]")
    if seed_scores:
        mean_of_r = float(np.mean(seed_scores))
        r_of_mean = pearson_ci(y_true, merged[GVP_COLUMN].to_numpy())["r"]
        print(
            f"  mean of per-seed r = {mean_of_r:+.3f} "
            f"(sd {np.std(seed_scores):.3f}, n={len(seed_scores)});  "
            f"r of the seed-mean prediction = {r_of_mean:+.3f}"
        )
        print("  (different quantities, per notebook.md - the report must say which it uses)")

    # ------------------------------------------- intervals and calibration
    print("\nIntervals and calibration ...")
    write("intervals", pd.DataFrame([
        {"model": name,
         **pearson_ci(y_true, merged[name].to_numpy()),
         **calibration_diagnostics(y_true, merged[name].to_numpy())}
        for name in all_models
    ]))

    # ------------------------------------------------------- stratification
    print("Stratifying by ligand novelty ...")
    stratified = []
    for name in all_models:
        frame = stratify_by_similarity(y_true, merged[name].to_numpy(), similarity)
        frame.insert(0, "model", name)
        stratified.append(frame)
    stratified = pd.concat(stratified, ignore_index=True)
    write("stratified", stratified)

    # ----------------------------------------------- the paired comparisons
    print(f"Paired bootstrap, {args.n_boot} resamples ...")
    rows = []
    for reference in references:
        for metric in ("pearson_r", "rmse"):
            rows.append({
                "reference": reference,
                "model": GVP_COLUMN,
                "metric": metric,
                "stratum": "all",
                **paired_bootstrap_delta(
                    y_true,
                    merged[reference].to_numpy(),
                    merged[GVP_COLUMN].to_numpy(),
                    metric=metric,
                    n_boot=args.n_boot,
                ),
            })

    # And on the novel-ligand stratum specifically, which is where the notebook
    # says an interaction model has to prove itself - the overall number is
    # inflated for every model by the ~14% of test ligands that appear verbatim
    # in training.
    novel = similarity < 0.4
    if novel.sum() >= 8:
        for reference in references:
            rows.append({
                "reference": reference,
                "model": GVP_COLUMN,
                "metric": "pearson_r",
                "stratum": "novel_ligands_sim_lt_0.4",
                **paired_bootstrap_delta(
                    y_true[novel],
                    merged[reference].to_numpy()[novel],
                    merged[GVP_COLUMN].to_numpy()[novel],
                    metric="pearson_r",
                    n_boot=args.n_boot,
                ),
            })

    paired = pd.DataFrame(rows)
    write("paired", paired)

    # --------------------------------------------------------- the verdict
    print(f"\n{'=' * 78}\nGVP vs each reference, paired on the same complexes\n{'=' * 78}")
    for stratum in paired["stratum"].unique():
        subset = paired[
            (paired["stratum"] == stratum) & (paired["metric"] == "pearson_r")
        ]
        n = int(novel.sum()) if stratum != "all" else len(merged)
        print(f"\n{stratum}  (n = {n})")
        for row in subset.itertuples():
            if row.significant:
                verdict = "GVP WINS" if row.b_better else "GVP LOSES"
            else:
                verdict = "not resolvable"
            print(
                f"  vs {row.reference:<26} delta r = {row.delta:+.3f}  "
                f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]   {verdict}"
            )

    print(
        "\nRead 'not resolvable' as the finding it is: with these sample sizes the\n"
        "two models cannot be separated, which is the Volkov result restated. It is\n"
        "not a null to be explained away, and it is not evidence they are equal -\n"
        "it is evidence the benchmark does not distinguish them."
    )

    print(f"\nWrote {len(written)} tables to results/tables/:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
