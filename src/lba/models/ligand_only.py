"""Ligand-only baselines.

The protein is stripped entirely: these models see nothing but the ligand. Under
the project's hypothesis they should nonetheless score well, because affinity
labels in PDBbind are substantially predictable from ligand identity alone
(Volkov et al. 2022). 

Four featurizations, in increasing order of how much they memorize:

* ``descriptors`` - 20 interpretable physicochemical properties. Size, polarity,
  flexibility. Cannot identify a specific ligand, only describe it.
* ``descriptors_full`` - every descriptor RDKit exposes. Not interpretable, but
  it is the honest ceiling on what ligand-only physicochemistry can do, which is
  the number the project's argument actually rests on.
* ``morgan`` - circular fingerprints. Substructure identity, so effectively a
  fuzzy lookup key for "which molecule is this".
* ``nearest_neighbour`` - not learned at all. Copies the affinity of the most
  Tanimoto-similar training ligand. This is Volkov's memorization probe: if it
  scores near the trained models, the task is retrieval rather than prediction.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# RDKit is loud about sanitization on PDBbind ligands; we count failures instead.
RDLogger.DisableLog("rdApp.*")

# An interpretable descriptor set: Lipinski's Rule of Five inputs, Veber's rules,
# and a few classical topological indices. We keep it for the interpretable story -
# but note it is NOT the better-performing choice. Measured on the 30% split
# (RF, 3 seeds): this set reaches test r = 0.467, the full set below reaches
# 0.516. Six of these twenty (MolWt, HeavyAtomMolWt, HeavyAtomCount, LabuteASA,
# MolMR, BertzCT) are pairwise correlated above r = 0.9 - they are all measuring
# size - so the set carries far less independent information than its width
# suggests. Report both; the full set is the honest ligand-only ceiling.
DESCRIPTORS = [
    "MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "MolLogP", "MolMR", "TPSA", "NumHeteroatoms", "NHOHCount", "NOCount",
    "NumSaturatedRings", "NumAliphaticRings", "HeavyAtomMolWt", "BertzCT",
    "BalabanJ", "LabuteASA",
]

# Every descriptor RDKit ships. ``_descList`` is the conventional accessor
# for this despite the underscore. Nine of them go non-finite on the
# ~15% of ligands that need relaxed valence parsing (qed and the BCUT2D_*
# family need a valid valence model) and ten are constant on this
# dataset; median imputation below handles both, and the extra ~198 carry real
# signal the curated set lacks - electrotopological state, connectivity indices,
# and partial-charge-weighted surface areas.
ALL_DESCRIPTORS = [name for name, _ in Descriptors._descList]

_DESCRIPTOR_FNS = {name: fn for name, fn in Descriptors._descList}
# The curated names must all exist in RDKit's registry, or a typo would silently
# produce a column of NaNs that imputation would then quietly fill.
_missing = [name for name in DESCRIPTORS if name not in _DESCRIPTOR_FNS]
if _missing:  # pragma: no cover - guards against an RDKit rename
    raise ImportError(f"RDKit {Chem.rdBase.rdkitVersion} has no descriptor(s): {_missing}")


def _relaxed_parse(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES that fails RDKit's valence model.

    PDBbind SMILES frequently write phosphates, sulfonates and carboxylates
    without formal charges, giving P with 6-7 bonds or C with 5. Strict parsing
    rejects these outright. We re-parse without sanitization and then sanitize
    everything *except* the valence/property check, which leaves ring perception
    and aromaticity intact - enough for descriptors and fingerprints.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return mol


def parse_smiles(
    smiles_list: list[str], allow_relaxed: bool = True
) -> tuple[list[Chem.Mol], np.ndarray, int]:
    """Parse SMILES, returning molecules, a validity mask, and a relaxed count.

    Dropping unparseable ligands is *not* neutral on this dataset: they skew
    roughly 0.8 log units weaker in affinity than parseable ones, so discarding
    them removes weak binders from training. Hence the relaxed fallback.

    :param allow_relaxed: if False, only strictly-valid SMILES are kept. Use for
        measuring how much the fallback changes a result.
    :returns: (mols, valid_mask, n_relaxed)
    """
    mols, valid = [], np.zeros(len(smiles_list), dtype=bool)
    n_relaxed = 0
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None and allow_relaxed:
            mol = _relaxed_parse(smiles)
            if mol is not None:
                n_relaxed += 1
        if mol is not None:
            mols.append(mol)
            valid[i] = True
    return mols, valid, n_relaxed


def descriptor_features(
    mols: list[Chem.Mol],
    names: list[str] = DESCRIPTORS,
    medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_mols, len(names)) physicochemical feature matrix, plus its medians.

    A few descriptors go non-finite on odd fragments (BalabanJ especially, and
    the whole BCUT2D family on relaxed-parsed ligands), so missing values are
    median-imputed rather than dropping the ligand.

    Those medians are a *fitted parameter*. Computing them per split would let
    val/test statistics into their own features, so the caller fits on train and
    passes the result back in for every other split:

        x_train, medians = descriptor_features(train_mols)
        x_test, _ = descriptor_features(test_mols, medians=medians)

    :param names: which descriptors to compute. ``DESCRIPTORS`` (interpretable)
        or ``ALL_DESCRIPTORS`` (the full RDKit set).
    :param medians: per-column medians fitted on the training split. If None,
        they are fitted from ``mols`` - correct only when these *are* the
        training molecules.
    :returns: (features, medians)
    """
    rows = []
    for mol in mols:
        row = []
        for name in names:
            try:
                value = _DESCRIPTOR_FNS[name](mol)
            except Exception:  # noqa: BLE001 - a bad descriptor is not fatal
                value = np.nan
            row.append(value)
        rows.append(row)
    features = np.asarray(rows, dtype=float).reshape(len(mols), len(names))

    if medians is None:
        medians = np.nanmedian(np.where(np.isfinite(features), features, np.nan), axis=0)
        # A column non-finite for every training ligand has no median to take.
        medians = np.where(np.isfinite(medians), medians, 0.0)
    elif medians.shape != (len(names),):
        raise ValueError(f"medians has shape {medians.shape}, expected {(len(names),)}")

    bad = ~np.isfinite(features)
    if bad.any():
        features[bad] = np.take(medians, np.where(bad)[1])
    return features, medians


def morgan_fingerprints(
    mols: list[Chem.Mol], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """(n_mols, n_bits) binary Morgan/ECFP fingerprint matrix."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    features = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, features[i])
    return features


def _bit_vectors(mols: list[Chem.Mol], radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    return [generator.GetFingerprint(mol) for mol in mols]


def nearest_neighbour_predict(
    train_mols: list[Chem.Mol],
    train_labels: np.ndarray,
    query_mols: list[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy the affinity of the most Tanimoto-similar training ligand(s).

    This is deliberately not a model. It is the retrieval baseline from Volkov
    et al.: if it approaches the trained models, the benchmark is rewarding
    lookup rather than any understanding of binding.

    :returns: (predictions, max_similarity_per_query). The similarity array is
        the diagnostic - high values mean the test ligand had a near-duplicate
        in training, which is leakage the split failed to remove.
    """
    train_fps = _bit_vectors(train_mols, radius, n_bits)
    query_fps = _bit_vectors(query_mols, radius, n_bits)
    train_labels = np.asarray(train_labels, dtype=float)

    predictions = np.empty(len(query_fps), dtype=float)
    similarities = np.empty(len(query_fps), dtype=float)
    for i, fingerprint in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))
        top = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        predictions[i] = train_labels[top].mean()
        similarities[i] = sims[top].max()
    return predictions, similarities
