#!/usr/bin/env python3
"""Verify the GVP model's prediction is invariant to rotation and translation.

    .venv/bin/python scripts/test_equivariance.py

Affinity is a scalar: rotate a complex in space and it binds exactly as
tightly. A model that does not respect that has to learn it from 3507 training
complexes, and what it will actually learn is the orientations in which
structures happen to have been deposited.

This is the one architectural claim in the project that can be *verified*
instead of argued, so it runs as a gate before any training. Three checks:

1. **Rotation.** Apply a random rotation to every coordinate, rebuild the graph,
   predict. The output must be unchanged to float tolerance.
2. **Translation.** Same, with a large random offset. ``build_complex_graph``
   centres coordinates and only ever uses differences, so this must hold too.
3. **Permutation.** Shuffle atom order. A graph network must not care, and this
   catches an indexing bug that rotation alone would not.

A note on tolerance: the model runs in float32 and the graph is rebuilt from
rotated coordinates, so ``radius_graph`` can legitimately include or exclude an
edge sitting exactly at the 5 A cutoff. A handful of complexes may therefore
differ by more than pure float noise for an honest reason. The check reports the
distribution rather than asserting on a single complex, and is run in float64 as
a cross-check where the tolerance can be tight.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lba.data.graph import build_complex_graph, graph_dims  # noqa: E402
from lba.data.lba import DEFAULT_RAW, load_raw  # noqa: E402
from lba.models.gvp import GVPAffinityModel  # noqa: E402


def random_rotation(seed: int) -> np.ndarray:
    """A uniformly random proper rotation matrix via QR of a Gaussian matrix.

    The sign correction is what makes it *proper* (determinant +1). Without it
    you get a random orthogonal matrix, half of which are reflections - and a
    reflection is a genuinely different molecule, so testing against one would
    be asserting something false about chirality.
    """
    generator = np.random.default_rng(seed)
    q, r = np.linalg.qr(generator.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def transform_item(item: dict, rotation: np.ndarray | None, offset: np.ndarray | None) -> dict:
    """Copy one raw item with its coordinates rotated and/or translated."""
    out = dict(item)
    for key in ("atoms_pocket", "atoms_ligand"):
        frame = item[key].copy()
        coords = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if rotation is not None:
            coords = coords @ rotation.T
        if offset is not None:
            coords = coords + offset
        frame[["x", "y", "z"]] = coords
        out[key] = frame
    return out


def permute_item(item: dict, seed: int) -> dict:
    """Copy one raw item with pocket and ligand atom order shuffled."""
    generator = np.random.default_rng(seed)
    out = dict(item)
    for key in ("atoms_pocket", "atoms_ligand"):
        frame = item[key]
        order = generator.permutation(len(frame))
        out[key] = frame.iloc[order].reset_index(drop=True)
    return out


def predict(model: GVPAffinityModel, item: dict, dtype: torch.dtype) -> float:
    """Build the graph for one item and run the model on it."""
    from torch_geometric.data import Batch

    graph = build_complex_graph(item, dtype=dtype)
    with torch.no_grad():
        return float(model(Batch.from_data_list([graph])).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument("--n", type=int, default=12, help="complexes to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max allowed absolute difference in predicted pKd (float32)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dims = graph_dims()
    # Untrained weights are the right thing to test: invariance is a property of
    # the architecture, not of the fit. Testing a trained checkpoint would
    # conflate the two.
    model = GVPAffinityModel(
        node_in_dims=(dims["node_s"], dims["node_v"]),
        edge_in_dims=(dims["edge_s"], dims["edge_v"]),
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")

    raw = load_raw(args.split, args.root)
    n = min(args.n, len(raw))
    print(f"testing {n} complexes from {args.split}\n")

    results: dict[str, list[float]] = {"rotation": [], "translation": [], "permutation": []}
    failures: list[str] = []

    for dtype, tolerance, label in ((torch.float32, args.tolerance, "float32"), (torch.float64, 1e-9, "float64")):
        model.to(dtype)
        deltas: dict[str, list[float]] = {k: [] for k in results}
        for index in range(n):
            item = raw[index]
            rotation = random_rotation(args.seed + index)
            offset = np.random.default_rng(args.seed + index).normal(scale=50.0, size=3)

            reference = predict(model, item, dtype)
            deltas["rotation"].append(
                abs(predict(model, transform_item(item, rotation, None), dtype) - reference)
            )
            deltas["translation"].append(
                abs(predict(model, transform_item(item, None, offset), dtype) - reference)
            )
            deltas["permutation"].append(
                abs(predict(model, permute_item(item, args.seed + index), dtype) - reference)
            )

        print(f"  {label}:")
        for check, values in deltas.items():
            worst = max(values)
            median = float(np.median(values))
            status = "PASS" if worst <= tolerance else "FAIL"
            if worst > tolerance:
                failures.append(f"{label}/{check}: max delta {worst:.2e} > {tolerance:.0e}")
            print(
                f"    {check:<12} max {worst:.3e}  median {median:.3e}  "
                f"(tol {tolerance:.0e})  {status}"
            )
            if label == "float32":
                results[check] = values
        print()

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nA rotation failure means geometry is leaking in as absolute "
            "coordinates somewhere. A permutation failure means an indexing bug. "
            "Do not train until this passes."
        )
        return 1

    print("All checks passed. Prediction is invariant to rotation, translation and atom ordering, " \
        "so the model cannot be reading the crystal frame.")
    

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
