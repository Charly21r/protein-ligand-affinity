"""Verify the environment is usable. Run after setup, or when something breaks.

    .venv/bin/python scripts/check_env.py
"""

import importlib
import platform
import sys

CHECKS = [
    ("torch", "__version__"),
    ("torch_geometric", "__version__"),
    ("torch_scatter", None),
    ("torch_cluster", None),
    ("atom3d", "__version__"),
    ("rdkit", "__version__"),
    ("Bio", "__version__"),
    ("py3Dmol", None),
    ("numpy", "__version__"),
    ("pandas", "__version__"),
    ("sklearn", "__version__"),
]


def main() -> int:
    failures = []
    print(f"python  {platform.python_version()}")
    if sys.version_info[:2] != (3, 11):
        failures.append("python is not 3.11 — atom3d and the PyG wheels expect it")

    for module, attr in CHECKS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, attr, "") if attr else ""
            print(f"{module:<16} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report anything
            print(f"{module:<16} FAILED: {exc}")
            failures.append(f"{module} does not import")

    # The compiled extensions are the fragile part: importing them is not
    # enough, they have to actually dispatch on the device we will train on.
    try:
        import torch
        import torch_cluster
        import torch_scatter

        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\ngpu             {name} ({total:.1f} GiB)")
        else:
            device = "cpu"
            print("\ngpu             none — falling back to CPU")

        x = torch.randn(8, 3, device=device)
        index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
        torch_scatter.scatter_mean(x, index, dim=0)
        torch_cluster.knn_graph(x, k=3)
        print(f"extensions      scatter + cluster OK on {device}")
    except Exception as exc:  # noqa: BLE001
        print(f"extensions      FAILED: {exc}")
        failures.append("torch_scatter/torch_cluster do not run")

    # atom3d rebuilds its serialized DataFrames only when the type tag in the
    # LMDB matches str(pd.DataFrame). pandas 3.0 changed that string, which
    # makes every atoms_* field load as a dict with no error raised at all.
    try:
        import pandas as pd

        tag = str(pd.DataFrame)
        if tag != "<class 'pandas.core.frame.DataFrame'>":
            print(f"\npandas          {pd.__version__} INCOMPATIBLE (tag: {tag})")
            failures.append(
                f"pandas {pd.__version__} breaks atom3d DataFrame deserialization "
                "silently - pin pandas<3.0"
            )
        else:
            print(f"\npandas          {pd.__version__} atom3d tag OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pandas check failed: {exc}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
