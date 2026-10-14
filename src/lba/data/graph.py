"""Graph construction for the interaction-aware model.

This is the first module in the project that lets a model see the ligand and the
pocket *at the same time*. Both baselines were deliberately blinded - one sees
only the drug, one only the protein - and the headline finding so far is that
neither blinding costs you much (ligand-only r = 0.516, size-free pocket-only
r = 0.477, statistically indistinguishable). The point of building a complex
graph is to test whether a model given the geometry between the two halves does
any better than a model given either half alone.

Three decisions here are load-bearing, and each one is a place where the
pipeline could silently cheat.

**Hydrogens are dropped.** They are ~52% of the atoms in these frames and were
added computationally, since X-ray crystallography does not resolve them. Any
element count over the raw frame is mostly a hydrogen count. The baselines drop
them (``models.pocket_only.summarize_pocket``) and so must this, or the
comparison is between different molecules.

**Coordinates never enter as features.** These structures sit in whatever
crystal frame they were deposited in, so an absolute position is a fact about
the PDB entry, not about the complex. Only distances and unit direction vectors
are used, which is what makes the rotation test in
``scripts/test_equivariance.py`` pass rather than merely approximately pass.

**The pocket is trimmed, and that trim is a confound.** ATOM3D already cut the
pocket by proximity to the ligand, which is the origin of the ``n_residues``
problem documented in ``models.pocket_only``: pocket size substantially *is*
ligand size (r = 0.695), and a linear fit on that one integer scores r = 0.576
on test, beating the entire 217-descriptor ligand model. Trimming further to
``pocket_cutoff`` does not remove that confound - nothing at this stage can,
because it is baked into the benchmark - but it does keep graphs small enough
for a 4 GiB card. Treat node count as a feature the model can read, and check
any score against the ``n_residues`` baseline rather than against
predict-the-mean.

**The neighbour cap must not bind.** ``torch_cluster.radius_graph`` truncates to
``max_num_neighbors`` by *atom index order*, not by distance, so a cap that
actually engages discards arbitrary close contacts and makes the graph depend on
the order atoms happen to appear in the deposited frame. An initial cap of 32 was
truncating a node in 77% of training complexes, which the permutation arm of
``scripts/test_equivariance.py`` caught. Over 300 training complexes the uncapped
maximum degree at a 5 A cutoff is 42, so the default is 48: above the observed
ceiling, costing ~9% more edges, and leaving the graph a true radius graph.
Raising ``edge_cutoff`` means re-measuring this.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from lba.data.lba import DEFAULT_CACHE, DEFAULT_RAW, load_raw

# Elements covering essentially all heavy atoms in the LBA frames: protein
# backbone and side chains, the common drug-like halogens, and the catalytic
# metals that appear as pocket residues. Anything else lands in an "other"
# bucket rather than being dropped, mirroring ``frac_other`` in the pocket
# baseline - an unrecognised element should stay visible, not vanish.
ELEMENTS = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE", "ZN", "MG", "MN", "FE", "CA", "NA", "K", "CU", "CO")
ELEMENT_INDEX = {element: i for i, element in enumerate(ELEMENTS)}
N_ELEMENT_FEATURES = len(ELEMENTS) + 1  # + "other"

# Edge provenance. Pocket-ligand edges are the only ones that carry interaction
# information at all, so they are labelled explicitly: it costs three dimensions
# and it makes the ablation cheap. Zeroing this channel, or deleting the cross
# edges outright, is the direct test of whether the model uses the interface or
# is reading the two halves separately.
EDGE_TYPES = ("pocket-pocket", "ligand-ligand", "pocket-ligand")
N_EDGE_TYPE_FEATURES = len(EDGE_TYPES)


def element_one_hot(elements: pd.Series) -> np.ndarray:
    """(n_atoms, N_ELEMENT_FEATURES) one-hot over :data:`ELEMENTS` plus "other".

    Element symbols in these frames are upper-case ("CL", not "Cl"), but that is
    not guaranteed across atom3d releases, so they are normalised here.
    """
    normalised = elements.astype(str).str.strip().str.upper()
    out = np.zeros((len(normalised), N_ELEMENT_FEATURES), dtype=np.float32)
    for row, symbol in enumerate(normalised):
        out[row, ELEMENT_INDEX.get(symbol, len(ELEMENTS))] = 1.0
    return out


def heavy_atoms(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop hydrogens, and anything with a non-finite coordinate.

    Non-finite coordinates are rare but do occur in deposited structures. They
    have to go before the radius graph is built: a single NaN position makes
    every distance involving that atom NaN, which propagates into the loss as a
    silent NaN rather than an error.
    """
    heavy = frame[frame["element"].astype(str).str.strip().str.upper() != "H"]
    coords = heavy[["x", "y", "z"]].to_numpy(dtype=float)
    return heavy[np.isfinite(coords).all(axis=1)]


def trim_pocket(
    pocket: pd.DataFrame, ligand: pd.DataFrame, cutoff: float
) -> pd.DataFrame:
    """Keep pocket atoms within ``cutoff`` A of any ligand atom.

    ATOM3D's pockets already have a cutoff applied, but it is generous: they run
    to ~650 atoms, and the median is ~170. Trimming to 8 A typically halves that,
    which is the difference between fitting a useful batch on a 4 GiB card and
    not. ``cutoff`` of ``inf`` keeps the pocket as atom3d shipped it.
    """
    if not np.isfinite(cutoff):
        return pocket
    if len(pocket) == 0 or len(ligand) == 0:
        return pocket

    # float64 regardless of the graph's own precision: this only decides which
    # atoms are kept, and rounding an atom sitting exactly on the cutoff in or
    # out would make the node set depend on coordinate precision.
    pocket_xyz = pocket[["x", "y", "z"]].to_numpy(dtype=np.float64)
    ligand_xyz = ligand[["x", "y", "z"]].to_numpy(dtype=np.float64)
    # (n_pocket, n_ligand) is at most ~650 x ~100 here, so the dense distance
    # matrix is cheaper than building a tree.
    deltas = pocket_xyz[:, None, :] - ligand_xyz[None, :, :]
    nearest = np.sqrt((deltas**2).sum(axis=-1)).min(axis=1)
    return pocket[nearest <= cutoff]


def rbf_expand(
    distances: torch.Tensor, n_bases: int = 16, cutoff: float = 5.0
) -> torch.Tensor:
    """Expand scalar distances onto ``n_bases`` Gaussian radial basis functions.

    A raw distance is a poor input to an MLP: the network has to carve it up
    before it can express "contact at 3.5 A but not 4.5 A". Gaussians evenly
    spaced over [0, cutoff] with width equal to their spacing do that carving
    up front, which is standard for this family of models.
    """
    centres = torch.linspace(0.0, cutoff, n_bases, device=distances.device)
    width = centres[1] - centres[0] if n_bases > 1 else torch.tensor(1.0)
    return torch.exp(-(((distances.unsqueeze(-1) - centres) / width) ** 2))


def build_complex_graph(
    item: dict,
    pocket_cutoff: float = 8.0,
    edge_cutoff: float = 5.0,
    max_neighbors: int = 48,
    n_rbf: int = 16,
    dtype: torch.dtype = torch.float32,
) -> Data:
    """One protein-ligand complex as a PyG graph.

    Node features are split the way a GVP expects: ``x`` holds rotation-invariant
    scalars (element one-hot + an is-ligand flag), and there are no initial node
    vector features, because a lone atom has no intrinsic orientation. All
    geometry enters through the edges, as a distance expansion (scalar) and a
    unit direction (vector).

    ``pos`` is kept on the graph so the equivariance test can rotate it and
    rebuild, but it is centred first and is never read as a feature.

    :param item: one raw atom3d LBA item, with ``atoms_pocket``, ``atoms_ligand``,
        ``id`` and ``scores``.
    :param pocket_cutoff: keep pocket atoms within this many A of the ligand.
    :param edge_cutoff: connect atoms closer than this many A.
    :param max_neighbors: cap on edges per node. Default 48, which is above the
        observed maximum and therefore never actually binds - see below.
    :param n_rbf: number of radial basis functions for the distance expansion.
    :param dtype: precision for coordinates and features. float32 for training;
        ``scripts/test_equivariance.py`` passes float64 so that its tight
        tolerance measures the architecture rather than the graph build's own
        rounding.
    :returns: a :class:`torch_geometric.data.Data` with ``x``, ``edge_index``,
        ``edge_s``, ``edge_v``, ``pos``, ``is_ligand``, ``y`` and ``pdb_id``.
    """
    from torch_cluster import radius_graph

    ligand = heavy_atoms(item["atoms_ligand"])
    pocket = trim_pocket(heavy_atoms(item["atoms_pocket"]), ligand, pocket_cutoff)

    if len(ligand) == 0:
        raise ValueError(f"{item.get('id')!r}: no heavy ligand atoms")
    if len(pocket) == 0:
        raise ValueError(f"{item.get('id')!r}: no pocket atoms within {pocket_cutoff} A")

    # Pocket first, then ligand, so is_ligand is a contiguous block. Nothing
    # depends on the order, but it makes the graphs far easier to inspect.
    elements = pd.concat([pocket["element"], ligand["element"]], ignore_index=True)
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    scalars = torch.from_numpy(element_one_hot(elements)).to(dtype)
    is_ligand = torch.cat([torch.zeros(len(pocket)), torch.ones(len(ligand))]).to(dtype)
    node_s = torch.cat([scalars, is_ligand.unsqueeze(-1)], dim=-1)

    coords = np.concatenate(
        [
            pocket[["x", "y", "z"]].to_numpy(dtype=numpy_dtype),
            ligand[["x", "y", "z"]].to_numpy(dtype=numpy_dtype),
        ]
    )
    pos = torch.from_numpy(coords)
    # Centring removes the crystal frame's origin. It does not affect any edge
    # feature - those are differences - but it keeps pos itself interpretable
    # and bounded, which matters when the equivariance test rotates it.
    pos = pos - pos.mean(dim=0, keepdim=True)

    edge_index = radius_graph(pos, r=edge_cutoff, max_num_neighbors=max_neighbors)
    if edge_index.numel() == 0:
        raise ValueError(f"{item.get('id')!r}: no edges within {edge_cutoff} A")

    source, target = edge_index
    displacement = pos[target] - pos[source]
    distances = displacement.norm(dim=-1)

    # Unit directions, not displacements: the magnitude is already carried by the
    # RBF expansion, and leaving it in the vector channel too would let the
    # model double-count it. Guard the division - radius_graph excludes
    # self-loops, so a zero distance means two atoms at identical coordinates,
    # which does happen in deposited structures.
    directions = displacement / distances.clamp(min=1e-6).unsqueeze(-1)

    ligand_source = is_ligand[source] > 0.5
    ligand_target = is_ligand[target] > 0.5
    edge_type = torch.zeros(len(distances), N_EDGE_TYPE_FEATURES, dtype=dtype)
    edge_type[~ligand_source & ~ligand_target, 0] = 1.0  # pocket-pocket
    edge_type[ligand_source & ligand_target, 1] = 1.0  # ligand-ligand
    edge_type[ligand_source ^ ligand_target, 2] = 1.0  # pocket-ligand, the interface

    edge_s = torch.cat([rbf_expand(distances, n_rbf, edge_cutoff), edge_type], dim=-1)
    edge_v = directions.unsqueeze(-2)  # (n_edges, 1, 3): one vector channel

    return Data(
        x=node_s,
        edge_index=edge_index,
        edge_s=edge_s,
        edge_v=edge_v,
        pos=pos,
        is_ligand=is_ligand,
        y=torch.tensor([float(item["scores"]["neglog_aff"])], dtype=dtype),
        pdb_id=str(item["id"]),
        n_pocket=int(len(pocket)),
        n_ligand=int(len(ligand)),
    )


def graph_cache_path(
    split: str,
    root: pathlib.Path = DEFAULT_RAW,
    identity: int = 30,
    cache_dir: pathlib.Path = DEFAULT_CACHE,
    pocket_cutoff: float = 8.0,
    edge_cutoff: float = 5.0,
) -> pathlib.Path:
    """Where the built graphs for one configuration live.

    The filename carries a hash of the root *and* the cutoffs. The root hash is
    for the reason given in ``load_records``: a different copy of the data must
    not hit a stale cache. The cutoffs are in there because they change the
    graphs completely, and a cache keyed only on the split would hand back 8 A
    graphs for a 10 A run without complaining.
    """
    tag = hashlib.sha1(
        f"{root.resolve()}|{pocket_cutoff}|{edge_cutoff}".encode()
    ).hexdigest()[:8]
    return cache_dir / f"lba{identity}_{split}_{tag}_graphs.pt"


def build_split_graphs(
    split: str,
    root: pathlib.Path = DEFAULT_RAW,
    identity: int = 30,
    cache_dir: pathlib.Path | None = DEFAULT_CACHE,
    pocket_cutoff: float = 8.0,
    edge_cutoff: float = 5.0,
    max_neighbors: int = 48,
    n_rbf: int = 16,
    rebuild: bool = False,
) -> list[Data]:
    """Build (or load) every graph for one split.

    Walking the LMDB deserializes the full protein frame for each item and takes
    minutes per pass, which is intolerable once per epoch. The whole split is
    built once and cached as a single ``.pt``; a split is only ~100 MB of graphs,
    so holding it in memory is cheaper than any lazy alternative. On Colab, point
    ``cache_dir`` at Drive so a killed session does not throw the work away.

    Items that cannot be built - no ligand atoms, no pocket atoms in range, no
    edges - are skipped and counted, following ``load_records``' convention that
    each failure mode stays visible rather than being silently merged.
    """
    cache = None
    if cache_dir is not None:
        cache = graph_cache_path(
            split, root, identity, cache_dir, pocket_cutoff, edge_cutoff
        )
        if cache.exists() and not rebuild:
            return torch.load(cache, weights_only=False)

    graphs, failures = [], []
    for item in load_raw(split, root, identity):
        try:
            graphs.append(
                build_complex_graph(
                    item, pocket_cutoff, edge_cutoff, max_neighbors, n_rbf
                )
            )
        except ValueError as error:
            failures.append(str(error))
    if failures:
        print(f"  [{split}] skipped {len(failures)} complexes:")
        for message in failures[:5]:
            print(f"    {message}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, cache)
    return graphs


def graph_dims(n_rbf: int = 16) -> dict[str, int]:
    """Feature dimensions, so the model does not hard-code them.

    Returned as a dict rather than a tuple because the GVP constructor takes
    ``(scalar, vector)`` pairs and mixing those up is an easy, quiet mistake.
    """
    return {
        "node_s": N_ELEMENT_FEATURES + 1,  # elements + is_ligand
        "node_v": 0,  # atoms have no intrinsic orientation
        "edge_s": n_rbf + N_EDGE_TYPE_FEATURES,
        "edge_v": 1,  # unit direction
    }
