"""Shortcut diagnostics.

These answer "*what* is the model using", which is the question this project is
actually about. A headline score cannot distinguish a model that learned
interaction physics from one that recognised the molecule or counted its atoms;
these probes can.

None of them are one-off. :func:`stratify_by_similarity` is the analysis to run
on the Phase 2 GVP-GNN (if its advantage over ligand-only appears specifically
on *novel* ligands, it is using the pocket; if it is flat, it is not), and
:func:`single_feature_scan` is what turned up both of the project's sharpest
findings so far - that MolMR alone reaches r = 0.50, and that a pocket residue
count alone reaches r = 0.58.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lba.eval.metrics import regression_metrics

DEFAULT_SIMILARITY_EDGES = (0.0, 0.4, 0.7, 0.9, 1.01)


def stratify_by_similarity(
    y_true,
    y_pred,
    similarity,
    edges: tuple[float, ...] = DEFAULT_SIMILARITY_EDGES,
    min_count: int = 8,
) -> pd.DataFrame:
    """Score a model separately on novel vs near-duplicate test ligands.

    The 30% split separates proteins, not ligands, so ligand leakage survives it
    (~14% of LBA test ligands appear verbatim in training). A model that is
    really doing retrieval scores well only where it has seen something similar;
    one that generalizes degrades gracefully. This makes that visible.

    :param similarity: per-test-complex maximum Tanimoto similarity to any
        training ligand - the second return value of
        ``ligand_only.nearest_neighbour_predict``.
    :param min_count: buckets smaller than this are reported but flagged, since
        a Pearson r over a handful of points is not worth reading.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    similarity = np.asarray(similarity, dtype=float).ravel()

    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (similarity >= low) & (similarity < high)
        if not mask.any():
            continue
        scores = regression_metrics(y_true[mask], y_pred[mask])
        rows.append({
            "bucket": f"{low:.1f}-{high:.1f}",
            "sim_low": low,
            "sim_high": high,
            "n": int(mask.sum()),
            "reliable": bool(mask.sum() >= min_count),
            "label_sd": float(y_true[mask].std()),
            **scores,
        })
    return pd.DataFrame(rows)


def within_ligand_variance(smiles, labels) -> tuple[pd.DataFrame, dict[str, float]]:
    """How much affinity varies for the *same* molecule in different proteins.

    This is the cleanest statement of why a ligand-only model must fail, and it
    needs no model at all - it is a property of the dataset. If one molecule
    binds two different proteins with different affinities, no function of the
    ligand alone can produce both numbers.

    The summary's ``oracle_rmse`` is the error of a *perfect* ligand-only
    predictor on the repeated ligands (the best it can do is each ligand's own
    mean). That is the irreducible cost of ignoring the protein, and the
    headroom an interaction-aware model has to claim.

    :returns: (per-ligand frame for repeated ligands, summary dict)
    """
    frame = pd.DataFrame({"smiles": list(smiles), "label": np.asarray(labels, dtype=float)})
    grouped = frame.groupby("smiles")["label"]
    per_ligand = grouped.agg(["count", "mean", "std", "min", "max"])
    per_ligand["range"] = per_ligand["max"] - per_ligand["min"]
    repeated = per_ligand[per_ligand["count"] > 1].sort_values("range", ascending=False)

    merged = frame.merge(grouped.mean().rename("ligand_mean"), on="smiles")
    is_repeated = merged["smiles"].isin(repeated.index)
    residual = (merged["label"] - merged["ligand_mean"])[is_repeated]

    summary = {
        "n_complexes": int(len(frame)),
        "n_distinct_ligands": int(len(per_ligand)),
        "n_repeated_ligands": int(len(repeated)),
        "n_complexes_repeated": int(repeated["count"].sum()),
        "median_within_ligand_sd": float(repeated["std"].median()) if len(repeated) else float("nan"),
        "mean_within_ligand_sd": float(repeated["std"].mean()) if len(repeated) else float("nan"),
        "median_within_ligand_range": float(repeated["range"].median()) if len(repeated) else float("nan"),
        "max_within_ligand_range": float(repeated["range"].max()) if len(repeated) else float("nan"),
        "oracle_rmse": float(np.sqrt((residual**2).mean())) if len(residual) else float("nan"),
        "overall_label_sd": float(frame["label"].std()),
    }
    return repeated.reset_index(), summary


def single_feature_scan(features, y, names, top: int | None = None) -> pd.DataFrame:
    """Correlation of each individual feature with the label.

    Run this before trusting any model: if one raw feature matches the model's
    score, the model has not added anything and the benchmark is rewarding that
    feature rather than the task. Spearman is reported alongside Pearson so a
    monotone-but-nonlinear feature is not missed.

    :param top: keep only the strongest ``top`` features by |pearson_r|.
    """
    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if features.shape[0] != y.shape[0]:
        raise ValueError(f"{features.shape[0]} rows of features vs {y.shape[0]} labels")
    if features.shape[1] != len(names):
        raise ValueError(f"{features.shape[1]} columns vs {len(names)} names")

    rows = []
    for i, name in enumerate(names):
        column = features[:, i]
        if np.std(column) < 1e-12:
            rows.append({"feature": name, "pearson_r": np.nan, "spearman_r": np.nan,
                         "constant": True})
            continue
        scores = regression_metrics(y, column)
        rows.append({
            "feature": name,
            # regression_metrics treats the column as a prediction; only its
            # correlations are meaningful here, since the units are arbitrary.
            "pearson_r": scores["pearson_r"],
            "spearman_r": scores["spearman_r"],
            "constant": False,
        })
    frame = pd.DataFrame(rows)
    frame = frame.reindex(frame["pearson_r"].abs().sort_values(ascending=False).index)
    return frame.head(top).reset_index(drop=True) if top else frame.reset_index(drop=True)


def collinear_pairs(features, names, threshold: float = 0.9) -> pd.DataFrame:
    """Feature pairs correlated above ``threshold``.

    A wide feature matrix can carry far less independent information than its
    width suggests. The curated 20-descriptor set in this project has 20 pairs
    above 0.9, six of them all measuring molecular size, which is why it
    underperforms the full RDKit set despite looking better curated.
    """
    features = np.asarray(features, dtype=float)
    keep = features.std(axis=0) > 1e-12
    correlation = np.corrcoef(features[:, keep], rowvar=False)
    kept_names = [n for n, k in zip(names, keep) if k]

    rows = [
        {"feature_a": kept_names[i], "feature_b": kept_names[j], "r": float(correlation[i, j])}
        for i in range(len(kept_names))
        for j in range(i + 1, len(kept_names))
        if abs(correlation[i, j]) >= threshold
    ]
    frame = pd.DataFrame(rows, columns=["feature_a", "feature_b", "r"])
    return frame.reindex(frame["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def smiles_quality_report(smiles, labels=None) -> dict[str, float]:
    """What the dataset's SMILES do and do not encode.

    ATOM3D's LBA SMILES were derived from 3D coordinates with bond orders
    inferred, which is lossy in ways that bound what any ligand-only model can
    do. Imported lazily so this module does not pull in RDKit unless used.

    ``affinity_skew`` is the mean-label difference between strictly-unparseable
    and strictly-parseable ligands. It is the justification for the relaxed
    valence fallback: if unparseable ligands are systematically weaker binders,
    dropping them biases the training set.
    """
    from rdkit import Chem, RDLogger

    from lba.models.ligand_only import _relaxed_parse

    RDLogger.DisableLog("rdApp.*")
    smiles = list(smiles)
    strict = np.array([Chem.MolFromSmiles(s) is not None for s in smiles])
    relaxed = np.array([
        ok or (_relaxed_parse(s) is not None) for s, ok in zip(smiles, strict)
    ])

    report = {
        "n": len(smiles),
        "n_strict_parse": int(strict.sum()),
        "n_needs_relaxed": int((~strict & relaxed).sum()),
        "n_unparseable": int((~relaxed).sum()),
        "frac_with_stereo": float(
            np.mean([("@" in s) or ("/" in s) or ("\\" in s) for s in smiles])
        ),
        "frac_with_aromatic_notation": float(
            np.mean([any(c in "cnops" and c.islower() for c in s) for s in smiles])
        ),
    }
    if labels is not None:
        labels = np.asarray(labels, dtype=float)
        if strict.any() and (~strict).any():
            report["mean_label_strict"] = float(labels[strict].mean())
            report["mean_label_unparseable"] = float(labels[~strict].mean())
            report["affinity_skew"] = float(labels[~strict].mean() - labels[strict].mean())
    return report


def single_feature_baseline(
    x_train, y_train, x_test, y_test, names, features: list[str] | None = None
) -> pd.DataFrame:
    """Fit a one-feature linear model per feature and score it on the test split.

    :func:`single_feature_scan` reports a feature's correlation *within one
    split*. This is the stronger and more alarming question: how well does that
    single feature, fitted on train, actually predict the held-out test set?

    It is the baseline every real model has to clear. On LBA the pocket residue
    count alone reaches test r = 0.576, which exceeds a 217-descriptor
    ligand-only model and matches published 3D deep models on this split - so
    "beats predict-the-mean" is far too low a bar to judge Phase 2 against.

    :param features: which feature names to fit. Defaults to all of ``names``.
    """
    from sklearn.linear_model import LinearRegression

    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    chosen = list(names) if features is None else list(features)

    rows = []
    for name in chosen:
        i = list(names).index(name)
        column_train, column_test = x_train[:, [i]], x_test[:, [i]]
        if np.std(column_train) < 1e-12:
            continue
        model = LinearRegression().fit(column_train, y_train)
        rows.append({
            "feature": name,
            **regression_metrics(y_test, model.predict(column_test)),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("pearson_r", ascending=False).reset_index(drop=True)
