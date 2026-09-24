"""Pocket-only baseline.

The ligand is stripped entirely: these models see nothing but the binding site.
This is the other half of Volkov et al.'s argument. ``ligand_only`` measures how
much affinity is predictable from the drug alone; this measures how much is
predictable from the protein alone.

What it is really measuring is **protein family identity**. A 20-dimensional
amino-acid profile of a pocket is close to a family fingerprint, so whatever
score this reaches is a direct measurement of the second shortcut. 

The confound, which drives the whole design here
------------------------------------------------
ATOM3D defined the pocket *by proximity to the ligand*, so a bigger ligand
touches more residues and yields a bigger pocket. Over the full training split,
hydrogens excluded (regenerate with ``scripts/analyze_baselines.py``, which
writes ``results/tables/analysis_pocket_size_confound.csv``)::

                        r with ligand heavy atoms   r with pKd
    n_residues                      0.695             0.469
    n_heavy_atoms                   0.663             0.447
    radius_of_gyration              0.745             0.403

Pocket size predicts affinity nearly as well as ligand size does (r = 0.488),
because it substantially *is* ligand size. A pocket-only model fed absolute
counts would score well while covertly measuring the ligand, which would
invalidate the comparison this baseline exists to support.

Worse, these features could not exist in the real prospective task. Computing
``n_residues`` requires a pocket already cut around a known ligand pose, which
is precisely what you do not have when asking "how tightly will this new
compound bind?". Fitted on train and scored on test, ``n_residues`` alone
reaches r = 0.576 / RMSE 1.387 - beating a 217-descriptor ligand-only model and
matching published 3D deep models on this split. The strongest single predictor
on this benchmark is an artifact of how the benchmark was built.

So two feature sets are provided:

* ``composition`` - amino-acid *fractions* and ion presence flags. Size-free by
  construction. This is the honest pocket-only number.
* ``composition_size`` - the above plus absolute size and shape. The gap between
  the two is how much apparent pocket signal is really ligand size in disguise.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from lba.data.lba import DEFAULT_CACHE, DEFAULT_RAW, load_raw

# The 20 standard amino acids. Anything else that is not a recognised ion goes
# into an "other" bucket rather than being silently dropped.
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

# Every ion seen as a pocket "residue" across the training split. Zinc in
# particular is catalytically decisive and strongly family-diagnostic, so these
# are kept rather than filtered out with the other non-standard residues.
IONS = ["CA", "CO", "CU", "FE", "FE2", "K", "MG", "MN", "NA", "SR", "ZN"]

COMPOSITION_FEATURES = (
    [f"frac_{a}" for a in AMINO_ACIDS] + ["frac_other"] + [f"has_{i}" for i in IONS]
)
# Absolute scale and shape. Every one of these correlates with ligand size via
# the pocket-definition confound above - that is exactly why they are separable.
SIZE_FEATURES = ["n_residues", "n_heavy_atoms", "radius_of_gyration", "extent_1", "extent_2", "extent_3"]

FEATURE_SETS = {
    "composition": COMPOSITION_FEATURES,
    "composition_size": COMPOSITION_FEATURES + SIZE_FEATURES,
}


def summarize_pocket(atoms: pd.DataFrame) -> dict[str, float]:
    """Reduce one ``atoms_pocket`` frame to a flat dict of features.

    Hydrogens are dropped first. They are ~52% of the atoms in these frames and
    were added computationally - X-ray crystallography does not resolve them -
    so leaving them in would turn any element count into a hydrogen count.

    Residues are identified by ``(chain, residue, insertion_code)``. The
    insertion code matters: protease and antibody numbering reuses residue
    numbers with different codes, and ignoring it silently merges distinct
    residues.
    """
    heavy = atoms[atoms["element"] != "H"]
    residues = heavy.groupby(["chain", "residue", "insertion_code"], sort=False)

    # One resname per residue, not per atom, so a large residue does not count
    # more than a small one.
    resnames = residues["resname"].first()
    counts = resnames.value_counts()

    amino = {a: int(counts.get(a, 0)) for a in AMINO_ACIDS}
    n_amino = sum(amino.values())
    n_ion = sum(int(counts.get(i, 0)) for i in IONS)
    n_other = int(counts.sum()) - n_amino - n_ion

    # Fractions are taken over amino-acid residues only, so that an ion or an
    # unrecognised residue cannot dilute the composition profile.
    denominator = max(n_amino, 1)
    features: dict[str, float] = {
        f"frac_{a}": amino[a] / denominator for a in AMINO_ACIDS
    }
    features["frac_other"] = n_other / denominator
    for ion in IONS:
        features[f"has_{ion}"] = float(counts.get(ion, 0) > 0)

    features["n_residues"] = float(len(resnames))
    features["n_heavy_atoms"] = float(len(heavy))

    # Shape has to be measured in a rotation-invariant way: these coordinates sit
    # in an arbitrary crystal frame, so per-axis bounding boxes would encode the
    # deposition frame rather than the pocket. Radius of gyration is invariant,
    # and the principal-axis extents are the invariant version of a bounding box.
    coords = heavy[["x", "y", "z"]].to_numpy(dtype=float)
    centred = coords - coords.mean(axis=0)
    features["radius_of_gyration"] = float(np.sqrt((centred**2).sum(axis=1).mean()))
    eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    for i, value in enumerate(sorted(np.maximum(eigenvalues, 0.0), reverse=True), start=1):
        features[f"extent_{i}"] = float(np.sqrt(value))
    return features


def pocket_table(
    split: str,
    root: pathlib.Path = DEFAULT_RAW,
    identity: int = 30,
    cache_dir: pathlib.Path | None = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Per-complex pocket features + label for one split, parquet-cached.

    Building this walks the LMDB, which deserializes the full protein frame for
    every item and takes minutes. The cache makes repeat runs instant. Its key
    carries a hash of the resolved root for the same reason ``load_records``'
    does - a different copy of the data must not hit a stale cache.
    """
    cache = None
    if cache_dir is not None:
        import hashlib

        tag = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:8]
        cache = cache_dir / f"lba{identity}_{split}_{tag}_pockets.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

    rows = []
    for item in load_raw(split, root, identity):
        row = {"pdb_id": str(item["id"]), "label": float(item["scores"]["neglog_aff"])}
        row.update(summarize_pocket(item["atoms_pocket"]))
        rows.append(row)
    frame = pd.DataFrame(rows)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
    return frame


def feature_matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    """(n_complexes, n_features) matrix for one named feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, have {sorted(FEATURE_SETS)}")
    columns = FEATURE_SETS[feature_set]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"pocket table is missing columns: {missing}")
    return frame[columns].to_numpy(dtype=float)
