"""Loading the ATOM3D LBA dataset.

The raw LMDB items are heavyweight (full protein, pocket and ligand atom
frames). For the ligand-only baseline we need very little of that, so
:func:`load_records` pulls out just the fields we use and caches the result.

Note the pandas constraint: atom3d rebuilds the serialized DataFrames only when
a type tag in the LMDB matches ``str(pd.DataFrame)``, which pandas 3.0 changed.
On pandas 3 every ``atoms_*`` field silently loads as a raw dict. See
``scripts/check_env.py``.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import pandas as pd
from atom3d.datasets import LMDBDataset

SPLITS = ("train", "val", "test")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class LigandRecord:
    """The ligand-only view of one complex."""

    pdb_id: str
    smiles: str
    label: float  # neglog_aff, i.e. pKd/pKi


def split_dir(split: str, root: pathlib.Path = DEFAULT_RAW, identity: int = 30) -> pathlib.Path:
    """Path to one split's LMDB directory.

    The canonical layout is ``split-by-sequence-identity-{identity}/data/{split}``,
    but it is not guaranteed: it has moved between atom3d releases, and the 60%
    split has to be downloaded into its own root because ``download_data.py``
    refuses a non-empty directory. So fall back to searching, the way
    ``scripts/smoke_test_lba.py`` does, rather than failing on a layout we could
    have found.

    :param identity: sequence-identity cutoff of the split, 30 or 60.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    canonical = root / f"split-by-sequence-identity-{identity}" / "data" / split
    if (canonical / "data.mdb").exists():
        return canonical

    for base in sorted(root.glob(f"*sequence-identity-{identity}*")):
        found = [p for p in base.rglob(split) if p.is_dir() and (p / "data.mdb").exists()]
        if found:
            # Shallowest match: the archive nests a second copy under some layouts.
            return min(found, key=lambda p: len(p.parts))

    raise FileNotFoundError(
        f"No LMDB for split {split!r} (identity {identity}) under {root}. "
        f"Looked for {canonical} and searched for *sequence-identity-{identity}*. "
        "Run scripts/download_data.py first."
    )


def load_raw(split: str, root: pathlib.Path = DEFAULT_RAW, identity: int = 30) -> LMDBDataset:
    """The untouched atom3d dataset, for when the atom frames are needed."""
    return LMDBDataset(str(split_dir(split, root, identity)))


def load_records(
    split: str,
    root: pathlib.Path = DEFAULT_RAW,
    identity: int = 30,
    cache_dir: pathlib.Path | None = DEFAULT_CACHE,
) -> list[LigandRecord]:
    """SMILES + affinity for one split, cached to parquet after the first read.

    Items whose SMILES is missing are dropped and counted; RDKit-unparseable
    SMILES are *kept* here and filtered at featurization time, so that the
    count of each failure mode stays visible rather than being silently merged.

    The cache filename carries a hash of the resolved ``root``, so pointing at a
    different copy of the data (a Drive mount on Colab, a second download for the
    60% split) cannot silently return the previous root's records.
    """
    cache = None
    if cache_dir is not None:
        root_tag = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:8]
        cache = cache_dir / f"lba{identity}_{split}_{root_tag}_ligands.parquet"
        if cache.exists():
            frame = pd.read_parquet(cache)
            return [LigandRecord(**row) for row in frame.to_dict("records")]

    dataset = load_raw(split, root, identity)
    records, missing = [], 0
    for item in dataset:
        smiles = item.get("smiles")
        if not smiles:
            missing += 1
            continue
        records.append(
            LigandRecord(
                pdb_id=str(item["id"]),
                smiles=str(smiles),
                label=float(item["scores"]["neglog_aff"]),
            )
        )
    if missing:
        print(f"  [{split}] dropped {missing} items with no SMILES")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.__dict__ for r in records]).to_parquet(cache, index=False)
    return records


def load_all_records(
    root: pathlib.Path = DEFAULT_RAW, identity: int = 30
) -> dict[str, list[LigandRecord]]:
    """All three splits at once."""
    return {split: load_records(split, root, identity) for split in SPLITS}
