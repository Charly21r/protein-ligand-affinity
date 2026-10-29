"""Build per-complex protein sequences for family clustering.

One FASTA record per complex, written to ``data/splits/<split>_seqs.fasta``.

Chain rule
----------
A complex may contain several chains, often the *same* chain deposited two or
four times (``1ugx`` holds four copies of an A+B pair), plus waters parked in a
pseudo-chain ``.``. Concatenating everything would hand MMseqs2 the same
sequence repeated, and a short peptide is not a protein family. So: among the
chains that actually contact the ligand (``atoms_pocket``), take the
**longest** one. One sequence per complex, which is what clustering expects.

Residue codes come from Biopython's *extended* PDB table, so modified residues
such as ``MSE`` (selenomethionine) map to their parent amino acid instead of
being dropped. Anything still unknown -- water, ions, sugars -- is skipped.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended

from lba.data.lba import SPLITS, load_raw

# Uppercase PDB-style keys ('ALA', 'MSE', ...) -> one-letter codes.
THREE_TO_ONE: dict[str, str] = {
    k.upper(): v for k, v in protein_letters_3to1_extended.items()
}

# A chain shorter than this is a peptide or a crystallisation artefact, not a
# protein family.
MIN_CHAIN_RESIDUES = 30

DEFAULT_OUT = pathlib.Path("data/splits")


def pocket_chains(atoms_pocket: pd.DataFrame) -> set[str]:
    """Chain IDs with at least one atom in the binding pocket."""
    return set(atoms_pocket["chain"].unique())


def chain_sequence(atoms_protein: pd.DataFrame, chain: str) -> str:
    """One-letter sequence for a single chain, in residue order.

    Atoms are collapsed to residues by ``(chain, residue, insertion_code)``,
    keeping file order (``sort=False``) so the sequence follows the chain.
    """
    sub = atoms_protein[atoms_protein["chain"] == chain]
    resnames = sub.groupby(
        ["chain", "residue", "insertion_code"], sort=False
    )["resname"].first()
    return "".join(THREE_TO_ONE[r] for r in resnames if r in THREE_TO_ONE)


def complex_sequence(item: dict) -> str:
    """Longest ligand-contacting protein chain, as a one-letter sequence."""
    atoms_protein = item["atoms_protein"]
    candidates = pocket_chains(item["atoms_pocket"])
    if not candidates:  # should not happen, but do not lose the complex
        candidates = set(atoms_protein["chain"].unique())

    seqs = [chain_sequence(atoms_protein, c) for c in sorted(candidates)]
    seqs = [s for s in seqs if len(s) >= MIN_CHAIN_RESIDUES]
    if not seqs:
        # Every contacting chain is short: fall back to the longest chain in
        # the structure, so the complex still gets a label.
        seqs = [
            chain_sequence(atoms_protein, c)
            for c in sorted(set(atoms_protein["chain"].unique()))
        ]
    return max(seqs, key=len) if seqs else ""


def build(split: str, limit: int | None = None) -> list[tuple[str, str]]:
    """``(pdb_id, sequence)`` for every complex in one split."""
    dataset = load_raw(split)
    n = len(dataset) if limit is None else min(limit, len(dataset))
    records: list[tuple[str, str]] = []
    dropped = 0
    for i in range(n):
        item = dataset[i]
        seq = complex_sequence(item)
        if not seq:
            dropped += 1
            continue
        records.append((str(item["id"]), seq))
    if dropped:
        print(f"  [{split}] dropped {dropped} complexes with no usable chain")
    return records


def write_fasta(records: list[tuple[str, str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for pdb_id, seq in records:
            fh.write(f">{pdb_id}\n{seq}\n")
    print(f"  wrote {len(records)} sequences -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=(*SPLITS, "all"))
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N complexes (dry run)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        print(f"[{split}]")
        records = build(split, args.limit)
        lengths = [len(s) for _, s in records]
        if lengths:
            median = sorted(lengths)[len(lengths) // 2]
            print(f"  n={len(records)} len min={min(lengths)} "
                  f"median={median} max={max(lengths)}")
        if args.limit is None:
            write_fasta(records, args.out / f"{split}_seqs.fasta")
        else:
            for pdb_id, seq in records[:5]:
                print(f"  >{pdb_id} ({len(seq)}) {seq[:60]}...")


if __name__ == "__main__":
    main()
