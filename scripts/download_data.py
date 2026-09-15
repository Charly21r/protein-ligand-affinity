"""Download the ATOM3D LBA dataset with the 30% sequence-identity split.

    .venv/bin/python scripts/download_data.py

Roughly 1 GB from Zenodo into data/raw/. Safe to re-run; skips if present.
"""

import argparse
import pathlib

from atom3d.datasets import download_dataset

DEFAULT_OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--split",
        default="sequence-identity-30",
        choices=["sequence-identity-30", "sequence-identity-60"],
        help="ATOM3D LBA split. The project's headline numbers use 30%%.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        print(f"{args.out} is not empty — skipping. Delete it to re-download.")
        return

    print(f"Downloading LBA ({args.split}) to {args.out} ...")
    download_dataset("lba", str(args.out), split=args.split)
    print("Done.")


if __name__ == "__main__":
    main()
