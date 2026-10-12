"""Geometric vector perceptron GNN over the protein-ligand complex.

This is the interaction-aware model the two baselines exist to be compared
against. Written from the paper (Jing et al., "Learning from Protein Structure
with Geometric Vector Perceptrons", ICLR 2021, plus the vector-gating variant
from the 2021 equivariant-macromolecule follow-up) rather than adapted from
``drorlab/gvp-pytorch``, which targets torch 1.x and has not been updated.

Why an equivariant architecture at all
--------------------------------------
Affinity is a scalar: rotate the complex and it does not change. A model that is
not built to respect that has to *learn* it from data, and with 3507 training
complexes it will instead learn the orientations in which structures happen to
have been deposited. Equivariance makes that impossible by construction, which
is the one claim here that can be verified rather than argued -
``scripts/test_equivariance.py`` rotates a complex and checks the prediction is
unchanged to float tolerance.

The representation
------------------
Every node and edge carries a pair ``(s, V)``: ``s`` is ``(n, n_scalar)`` of
rotation-invariant features, ``V`` is ``(n, n_vector, 3)`` of features that
rotate with the input. A :class:`GVP` is the one primitive that mixes them while
keeping that contract: scalars may read vector *magnitudes* (invariant), and
vectors are only ever produced by linear combination of other vectors (so they
rotate correctly). Nothing else in the layer stack is allowed to touch the
vector channels.

Node vectors start as zeros, because a lone atom has no intrinsic orientation -
you cannot manufacture a direction from an element type without breaking
equivariance. All geometry enters through the edge unit vectors built in
``lba.data.graph``, and the first round of message passing is what fills the
node vector channels in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

# (n_scalar, n_vector) throughout. Passing these the wrong way round is the
# easiest quiet mistake in this file, so every signature annotates it.
Dims = tuple[int, int]


def _norm(vectors: torch.Tensor, axis: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 norm that is safe to differentiate at zero.

    ``torch.norm`` has an infinite gradient at the origin, and node vectors here
    start at exactly zero, so the first backward pass would produce NaNs. Adding
    eps inside the square root costs nothing and removes that.
    """
    return torch.sqrt(torch.clamp((vectors**2).sum(dim=axis, keepdim=False), min=eps))


class GVP(nn.Module):
    """Geometric vector perceptron: ``(s, V) -> (s', V')``.

    The scalar path sees vector magnitudes, which are invariant, so it may use a
    normal nonlinearity. The vector path is a bare linear recombination, gated by
    a sigmoid computed from the *scalars* - a per-channel scale, which commutes
    with rotation, rather than any per-component transform, which would not.

    :param in_dims: ``(n_scalar, n_vector)`` on the way in.
    :param out_dims: ``(n_scalar, n_vector)`` on the way out.
    :param h_dim: width of the intermediate vector projection; defaults to
        ``max(in_vector, out_vector)``.
    :param vector_gate: use vector gating. The paper's later variant, and
        clearly better in their ablations; exposed only so it can be turned off
        for a sanity check.
    """

    def __init__(
        self,
        in_dims: Dims,
        out_dims: Dims,
        h_dim: int | None = None,
        activation: type[nn.Module] = nn.ReLU,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate

        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            # Scalars read the magnitudes of the projected vectors alongside
            # their own features: this is the only channel by which geometry
            # reaches the invariant path.
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if vector_gate:
                    self.wg = nn.Linear(self.so, self.vo)
        else:
            # No input vectors: the layer degenerates to a plain MLP, and any
            # requested output vectors are necessarily zero. That is not a bug
            # to work around - it is equivariance refusing to invent a direction
            # out of scalars.
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act = activation() if activation is not None else None

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """:param x: ``(s, V)`` with ``V`` shaped ``(..., n_vector, 3)``, or just
            ``s`` when ``in_dims`` has no vector channels.
        """
        if self.vi:
            s, v = x
            # (..., n_vector, 3) -> (..., 3, n_vector) so the channel mix is the
            # last axis, then back. The 3-axis is never mixed.
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)
            s_out = self.ws(torch.cat([s, _norm(vh)], dim=-1))
            if self.vo:
                v_out = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
                if self.vector_gate:
                    gate = torch.sigmoid(self.wg(s_out)).unsqueeze(-1)
                    v_out = v_out * gate
                else:
                    # Ungated variant scales by the vector's own magnitude.
                    v_out = v_out * torch.sigmoid(_norm(v_out)).unsqueeze(-1)
        else:
            s = x[0] if isinstance(x, tuple) else x
            s_out = self.ws(s)
            if self.vo:
                v_out = torch.zeros(
                    *s_out.shape[:-1], self.vo, 3, device=s_out.device, dtype=s_out.dtype
                )

        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        return (s_out, v_out) if self.vo else s_out


class LayerNorm(nn.Module):
    """LayerNorm for a ``(s, V)`` pair.

    Scalars get a standard LayerNorm. Vectors cannot: subtracting a mean or
    rescaling per component would break rotation equivariance. Instead they are
    divided by the root-mean-square vector norm, which is a single invariant
    scalar per node, so the direction of every vector survives untouched.
    """

    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x[0] if isinstance(x, tuple) else x)
        s, v = x
        rms = torch.sqrt(
            torch.clamp((v**2).sum(dim=-1).mean(dim=-1, keepdim=True), min=1e-8)
        )
        return self.scalar_norm(s), v / rms.unsqueeze(-1)


class Dropout(nn.Module):
    """Dropout for a ``(s, V)`` pair.

    Vector channels are dropped whole - one Bernoulli draw per channel, applied
    to all three components - because zeroing individual components would,
    again, not commute with rotation.
    """

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate
        self.scalar_dropout = nn.Dropout(rate)

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        if not isinstance(x, tuple):
            return self.scalar_dropout(x)
        s, v = x
        mask = torch.bernoulli(
            (1 - self.rate) * torch.ones(*v.shape[:-1], 1, device=v.device)
        )
        return self.scalar_dropout(s), v * mask / (1 - self.rate)


class GVPConv(nn.Module):
    """One round of message passing in GVP space.

    A message is built from the source node, the destination node and the edge,
    all as ``(s, V)`` pairs, passed through a small GVP stack, and **mean**-
    aggregated over neighbours.

    Mean, not sum, and that choice is not cosmetic here. Node degree in these
    graphs tracks pocket size, and pocket size is the documented confound in this
    project - it correlates with ligand heavy-atom count at r = 0.695 and a
    linear fit on residue count alone scores r = 0.576 on test. Sum aggregation
    would hand the model that confound through the back door, as a magnitude
    proportional to degree, on top of whatever it reads from the features. Mean
    aggregation does not remove the confound (node count is still implicitly
    available) but it stops the architecture from amplifying it.
    """

    def __init__(
        self,
        node_dims: Dims,
        edge_dims: Dims,
        n_layers: int = 3,
        activation: type[nn.Module] = nn.ReLU,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.node_dims = node_dims
        si, vi = node_dims
        se, ve = edge_dims
        # Source and destination node features plus the edge features.
        message_in: Dims = (2 * si + se, 2 * vi + ve)

        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(GVP(message_in, node_dims, activation=None, vector_gate=vector_gate))
        else:
            layers.append(GVP(message_in, node_dims, activation=activation, vector_gate=vector_gate))
            for _ in range(n_layers - 2):
                layers.append(GVP(node_dims, node_dims, activation=activation, vector_gate=vector_gate))
            layers.append(GVP(node_dims, node_dims, activation=None, vector_gate=vector_gate))
        self.message = nn.Sequential(*layers)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        source, target = edge_index
        message_s = torch.cat([s[source], edge_s, s[target]], dim=-1)
        message_v = torch.cat([v[source], edge_v, v[target]], dim=-2)
        out_s, out_v = self.message((message_s, message_v))
        # dim_size pinned to the node count: a node with no incoming edges must
        # still get a row, or the graph silently loses nodes.
        return (
            scatter_mean(out_s, target, dim=0, dim_size=s.shape[0]),
            scatter_mean(out_v, target, dim=0, dim_size=s.shape[0]),
        )


class GVPConvLayer(nn.Module):
    """Message passing + pointwise feedforward, each residual and normed.

    The standard transformer-style block, in GVP space: update from neighbours,
    add and norm, then a per-node feedforward, add and norm.
    """

    def __init__(
        self,
        node_dims: Dims,
        edge_dims: Dims,
        n_message_layers: int = 3,
        n_feedforward: int = 2,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = GVPConv(
            node_dims, edge_dims, n_message_layers, activation, vector_gate
        )
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(dropout) for _ in range(2)])

        si, vi = node_dims
        wide: Dims = (4 * si, 2 * vi)
        feedforward: list[nn.Module] = []
        if n_feedforward == 1:
            feedforward.append(GVP(node_dims, node_dims, activation=None, vector_gate=vector_gate))
        else:
            feedforward.append(GVP(node_dims, wide, activation=activation, vector_gate=vector_gate))
            for _ in range(n_feedforward - 2):
                feedforward.append(GVP(wide, wide, activation=activation, vector_gate=vector_gate))
            feedforward.append(GVP(wide, node_dims, activation=None, vector_gate=vector_gate))
        self.feedforward = nn.Sequential(*feedforward)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        update_s, update_v = self.conv(s, v, edge_index, edge_s, edge_v)
        update_s, update_v = self.dropout[0]((update_s, update_v))
        s, v = self.norm[0]((s + update_s, v + update_v))

        update_s, update_v = self.feedforward((s, v))
        update_s, update_v = self.dropout[1]((update_s, update_v))
        return self.norm[1]((s + update_s, v + update_v))


class GVPAffinityModel(nn.Module):
    """Complex graph -> predicted pK_d.

    The output is a single scalar read off **invariant** features only: the
    vector channels do all the geometric work inside the network and are
    discarded before the head. That is what makes the prediction rotation
    *invariant* while the internals are rotation *equivariant*, and it is the
    property ``scripts/test_equivariance.py`` checks.

    Readout pools node scalars by mean over each graph. ``ligand_readout``
    additionally pools the ligand atoms on their own and concatenates: the
    interface is only ~12% of edges in a typical graph, so a plain global mean
    is dominated by bulk pocket atoms, and giving the head a ligand-local view
    costs little. It does not change what information is available, only how
    easily the head can reach it.

    :param node_in_dims: node ``(s, V)`` dims from ``lba.data.graph.graph_dims``.
    :param edge_in_dims: edge ``(s, V)`` dims, likewise.
    :param node_h_dims: hidden node dims. 16 vector channels against 100 scalars
        follows the paper's ratio.
    :param n_layers: message-passing rounds. Three gives a receptive field of
        ~15 A at a 5 A edge cutoff, which comfortably spans a binding site.
    """

    def __init__(
        self,
        node_in_dims: Dims,
        edge_in_dims: Dims,
        node_h_dims: Dims = (100, 16),
        edge_h_dims: Dims = (32, 1),
        n_layers: int = 3,
        dropout: float = 0.1,
        ligand_readout: bool = True,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.node_h_dims = node_h_dims
        self.ligand_readout = ligand_readout

        # Embed to hidden dims. The node embedding has no input vectors, so it
        # emits zero vectors by construction - see GVP above.
        self.node_embedding = nn.Sequential(
            LayerNorm(node_in_dims),
            GVP(node_in_dims, node_h_dims, activation=None, vector_gate=vector_gate),
        )
        self.edge_embedding = nn.Sequential(
            LayerNorm(edge_in_dims),
            GVP(edge_in_dims, edge_h_dims, activation=None, vector_gate=vector_gate),
        )
        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_h_dims, edge_h_dims, dropout=dropout, vector_gate=vector_gate
            )
            for _ in range(n_layers)
        )
        # Collapse to scalars only: vectors are dropped here, deliberately.
        self.to_scalars = nn.Sequential(
            LayerNorm(node_h_dims),
            GVP(node_h_dims, (node_h_dims[0], 0), activation=None, vector_gate=vector_gate),
        )

        width = node_h_dims[0] * (2 if ligand_readout else 1)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        """:param batch: a PyG ``Batch`` of graphs from ``lba.data.graph``.
        :returns: ``(n_graphs,)`` predicted pK_d.
        """
        # batch.x is a bare scalar tensor: node vector channels are zero on input.
        s, v = self.node_embedding(batch.x)
        edge_s, edge_v = self.edge_embedding((batch.edge_s, batch.edge_v))

        for layer in self.layers:
            s, v = layer(s, v, batch.edge_index, edge_s, edge_v)
        s = self.to_scalars((s, v))

        graph_index = getattr(batch, "batch", None)
        if graph_index is None:
            graph_index = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        n_graphs = int(graph_index.max()) + 1

        pooled = scatter_mean(s, graph_index, dim=0, dim_size=n_graphs)
        if self.ligand_readout:
            mask = batch.is_ligand > 0.5
            ligand_pooled = scatter_mean(
                s[mask], graph_index[mask], dim=0, dim_size=n_graphs
            )
            pooled = torch.cat([pooled, ligand_pooled], dim=-1)
        return self.head(pooled).squeeze(-1)
