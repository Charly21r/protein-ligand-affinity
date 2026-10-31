"""Protein-family labels for the adversarial discriminator.

``scripts/make_family_labels.py`` writes ``data/splits/train_families.csv``
with one row per training complex. This module turns that into the per-batch
target tensor the discriminator needs.

Labels are attached by ``pdb_id`` at training time rather than baked into the
graph cache, so switching K (or the identity threshold) does not mean
re-featurising 3507 complexes.

Complexes outside the top-K families carry ``MASK`` (-1), which
``nn.CrossEntropyLoss(ignore_index=-1)`` drops from the loss. Any complex
missing from the CSV is treated the same way.
"""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Sequence

import pandas as pd
import torch

MASK = -1
DEFAULT_LABELS = pathlib.Path("data/splits/train_families.csv")


@functools.lru_cache(maxsize=4)
def load_family_labels(path: str | pathlib.Path = DEFAULT_LABELS) -> dict[str, int]:
    """``pdb_id -> family_id``, cached so every epoch does not re-read the CSV."""
    df = pd.read_csv(path)
    return dict(zip(df["pdb_id"].astype(str), df["family_id"].astype(int)))


def n_families(labels: dict[str, int]) -> int:
    """Number of real classes, i.e. excluding the mask."""
    return max(labels.values()) + 1


def family_targets(
    pdb_ids: Sequence[str],
    labels: dict[str, int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``(n_graphs,)`` int64 targets for one batch, ``MASK`` where unlabelled."""
    return torch.tensor(
        [labels.get(str(p), MASK) for p in pdb_ids],
        dtype=torch.long,
        device=device,
    )
