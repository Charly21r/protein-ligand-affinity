"""Turn MMseqs2 clusters into discriminator labels.

Reads ``fam30_cluster.tsv`` (``representative <TAB> member``) and writes one
row per training complex with its family label.

Rare families are **masked**, not merged into an "other" class. At 30% identity
half of the 796 clusters are singletons, and folding the tail into one class
would make "other" the majority class (55.7% at K=20) -- a discriminator could
then score well without learning anything about families. Masked complexes get
``family_id = -1``: they still train the affinity head, but are excluded from
the adversarial loss (use ``ignore_index=-1`` in the cross-entropy).

The reported majority-class rate is over the *labelled* complexes only, since
that is the chance level the discriminator has to beat.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

DEFAULT_TSV = pathlib.Path("data/splits/fam30_cluster.tsv")
DEFAULT_OUT = pathlib.Path("data/splits/train_families.csv")
MASK = -1


def label_families(tsv: pathlib.Path, top_k: int) -> pd.DataFrame:
    """``pdb_id, cluster_rep, cluster_size, family_id`` for every complex."""
    df = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "pdb_id"])
    sizes = df.groupby("rep").size().sort_values(ascending=False)

    # Deterministic ids 0..K-1, largest cluster first; ties broken by name so
    # a rerun cannot silently renumber the classes.
    order = sizes.reset_index(name="n").sort_values(
        ["n", "rep"], ascending=[False, True]
    )
    kept = order.head(top_k)["rep"].tolist()
    family_id = {rep: i for i, rep in enumerate(kept)}

    out = df.copy()
    out["cluster_size"] = out["rep"].map(sizes)
    out["family_id"] = out["rep"].map(family_id).fillna(MASK).astype(int)
    return out.rename(columns={"rep": "cluster_rep"})[
        ["pdb_id", "cluster_rep", "cluster_size", "family_id"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=pathlib.Path, default=DEFAULT_TSV)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = label_families(args.tsv, args.top_k)
    labelled = out[out.family_id != MASK]
    counts = labelled.family_id.value_counts()

    print(f"complexes          {len(out)}")
    print(f"labelled           {len(labelled)} ({len(labelled) / len(out):.1%})")
    print(f"masked (-1)        {len(out) - len(labelled)}")
    print(f"classes            {args.top_k}")
    print(f"majority baseline  {counts.iloc[0] / len(labelled):.4f}"
          f"  <- chance level for discriminator accuracy")
    print(f"class sizes        {counts.sort_index().tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
