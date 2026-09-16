"""Verify the ATOM3D LBA dataset loads and looks the way it should.

    .venv/bin/python scripts/smoke_test_lba.py

Checks split sizes, the per-item schema, the atoms DataFrame, and the affinity
distribution (pKd/pKi should span roughly 2-12). Works locally and on Colab,
pass --root for a non-default path.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED = {"train": 3507, "val": 466, "test": 490}  # ATOM3D LBA, 30% identity


def find_split_dirs(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Locate train/val/test LMDB dirs without hardcoding the archive layout."""
    found: dict[str, pathlib.Path] = {}
    for split in ("train", "val", "test"):
        matches = [
            p for p in root.rglob(split) if p.is_dir() and (p / "data.mdb").exists()
        ]
        if matches:
            found[split] = min(matches, key=lambda p: len(p.parts))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"{args.root} does not exist. Run scripts/download_data.py first.")
        return 1

    splits = find_split_dirs(args.root)
    missing = {"train", "val", "test"} - splits.keys()
    if missing:
        print(f"Could not find LMDB dirs for: {sorted(missing)}")
        print(f"Looked under {args.root}. Is the archive extracted?")
        return 1

    problems: list[str] = []
    datasets = {}

    print("=== Splits ===")
    for split, path in splits.items():
        dataset = LMDBDataset(str(path))
        datasets[split] = dataset
        expected = EXPECTED[split]
        flag = "" if len(dataset) == expected else f"  <-- expected {expected}"
        print(f"{split:<6} {len(dataset):>5}{flag}")
        print(f"       {path.relative_to(args.root)}")
        if len(dataset) != expected:
            problems.append(f"{split} has {len(dataset)} items, expected {expected}")

    item = datasets["train"][0]
    print("\n=== Item schema (train[0]) ===")
    print(f"keys: {sorted(item.keys())}")
    print(f"id:   {item.get('id')}")

    for key in ("atoms_pocket", "atoms_ligand"):
        if key not in item:
            problems.append(f"train[0] is missing '{key}'")
            continue
        df = item[key]
        print(f"\n--- {key}: {df.shape[0]} atoms ---")
        print(f"columns:  {list(df.columns)}")
        print(f"elements: {sorted(df['element'].unique())}")
        for axis in ("x", "y", "z"):
            if axis not in df.columns:
                problems.append(f"{key} has no '{axis}' column")

    # The label. Everything downstream regresses against this.
    print("\n=== Affinity labels ===")
    for split, dataset in datasets.items():
        values = np.array([d["scores"]["neglog_aff"] for d in dataset])
        print(
            f"{split:<6} n={len(values):>5}  "
            f"min={values.min():.2f}  max={values.max():.2f}  "
            f"mean={values.mean():.2f}  std={values.std():.2f}"
        )
        if split == "train":
            if not (1.0 < values.min() < 4.0 and 9.0 < values.max() < 14.0):
                problems.append(
                    f"train affinities span {values.min():.2f}-{values.max():.2f}, "
                    "expected roughly 2-12 — is this really pKd/pKi?"
                )
            # Std of the training label is the baseline any model must beat:
            # predicting the mean gives RMSE == std and r == 0.
            print(f"\n       Predict-the-mean baseline: RMSE = {values.std():.3f}, r = 0")

    # Pocket sizes drive memory use, which is the binding constraint on 4 GiB.
    sizes = np.array([len(datasets["train"][i]["atoms_pocket"]) for i in range(200)])
    print(
        f"\n=== Pocket size (first 200 train) ===\n"
        f"min={sizes.min()}  median={int(np.median(sizes))}  "
        f"max={sizes.max()}  mean={sizes.mean():.0f}"
    )

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nLBA dataset OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
