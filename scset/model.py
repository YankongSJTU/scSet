# -*- coding: utf-8 -*-
"""scSet: a permutation-invariant set encoder over foundation-model cell
embeddings (terminal recipe of the paper's systematic dissection).

Architecture (d=256 default):
  Linear(in_dim -> d)
  GAT x2 over the kNN cell graph (k=15, cosine, built in embedding space;
         4 attention heads, concat -> width preserved)
  LayerNorm
  [TransformerEncoderLayer  -- only in the scSet-Contrastive prototype]
  dual permutation-invariant pooling:
      mean head  (d)  ||  attention head over 4 learnable queries (4d)
  MLP 5d -> 2d -> d -> d  (GELU / LayerNorm / dropout), L2-normalised output

Training head: Linear(d -> n_classes); cross-entropy (scSet) or supervised
contrastive (scSet-Contrastive, the initial prototype).

The GAT layer is implemented in pure torch; if torch_geometric is installed
the original GATConv is used instead (bit-compatible with the paper runs).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional, matches the paper implementation exactly
    from torch_geometric.nn import GATConv as _PyGGATConv
    from torch_geometric.utils import dense_to_sparse as _pyg_dense_to_sparse
    WITH_PYG = True
except Exception:  # pragma: no cover - fallback path
    _PyGGATConv = None
    _pyg_dense_to_sparse = None
    WITH_PYG = False


# --------------------------------------------------------------- kNN graph
def knn_edge_index(x: torch.Tensor, k: int = 15) -> torch.Tensor:
    """kNN graph (cosine) over a cell set; returns edge_index [2, E]."""
    n = x.shape[0]
    kk = min(k, n - 1)
    xn = F.normalize(x, dim=1)
    sim = xn @ xn.T
    sim.fill_diagonal_(-2.0)
    idx = sim.topk(kk, dim=1).indices           # (n, kk)
    mask = torch.zeros(n, n, dtype=torch.bool, device=x.device)
    mask.scatter_(1, idx, True)
    mask = mask | mask.T                          # symmetrise
    if WITH_PYG:
        return _pyg_dense_to_sparse(mask)[0]
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


# --------------------------------------------------------- pure-torch GAT
class PureGATLayer(nn.Module):
    """Minimal GATv2-style layer over a given edge set (fallback for PyG).

    h_i' = ELU( sum_j a_ij * W h_j ),  a_ij = softmax_j( LeakyReLU( s^T [W h_i
    || W h_j] ) ) per head; heads concatenated (width preserved when
    out_dim * heads == in_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.att, gain=1.414)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        H, D = self.heads, self.out_dim
        h = self.W(x).view(-1, H, D)                     # (n, H, D)
        src, dst = edge_index[0], edge_index[1]
        hs, hd = h[src], h[dst]                          # (E, H, D)
        g = torch.cat([hs, hd], dim=-1)                  # (E, H, 2D)
        alpha = (g * self.att).sum(-1)                   # (E, H)
        alpha = self.leaky(alpha)
        # per-destination softmax over incoming edges
        alpha = alpha - alpha.max()
        e = torch.exp(alpha)
        denom = torch.zeros(H, x.shape[0], device=x.device)
        denom.index_add_(1, dst, e)
        a = e / (denom[:, dst] + 1e-16)                  # (E, H)
        out = torch.zeros(x.shape[0], H, D, device=x.device)
        out.index_add_(0, dst, hs * a.unsqueeze(-1))
        return F.elu(out.reshape(-1, H * D))


class GATLayer(nn.Module):
    """Dispatches to torch_geometric GATConv when available (paper-exact)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.use_pyg = WITH_PYG
        if self.use_pyg:
            self.conv = _PyGGATConv(in_dim, out_dim, heads=heads, concat=True)
        else:
            self.conv = PureGATLayer(in_dim, out_dim, heads=heads)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


# ------------------------------------------------------------- the encoder
class scSet(nn.Module):
    """scSet set encoder.

    use_transformer=False -> scSet (terminal recipe: no Transformer block)
    use_transformer=True  -> the full architecture of the initial prototype
                             (scSet-Contrastive / scSet-CE family)
    """

    def __init__(self, input_dim: int, d: int = 256, heads: int = 4, k: int = 15,
                 use_transformer: bool = False, n_classes: int = 10,
                 n_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(input_dim, d)
        self.gat1 = GATLayer(d, d // heads, heads=heads)
        self.gat2 = GATLayer(d, d // heads, heads=heads)
        self.ln = nn.LayerNorm(d)
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=4 * d, batch_first=True,
                dropout=dropout, activation="gelu")
        self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear((1 + n_queries) * d, 2 * d), nn.GELU(),
            nn.LayerNorm(2 * d), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None):
        """x: (n_cells, input_dim) for ONE sample; returns (z, logits)."""
        if edge_index is None:
            edge_index = knn_edge_index(x, self.k)
        h = self.proj(x)
        h = F.gelu(self.gat1(h, edge_index))
        h = self.gat2(h, edge_index)
        h = self.ln(h)
        if self.use_transformer:
            h = self.transformer(h.unsqueeze(0)).squeeze(0)
        mean_head = h.mean(dim=0).flatten()                        # (d,)
        attn_head, _ = self.mha(self.queries.expand(1, -1, -1),
                                h.unsqueeze(0), h.unsqueeze(0))    # (1, q, d)
        pooled = torch.cat([mean_head, attn_head.squeeze(0).flatten()])[None]
        z = F.normalize(self.mlp(pooled), dim=1)
        return z, self.head(z)

    def embed_sample(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.forward(x)
        return z.squeeze(0)


# ----------------------------------------------------------------- losses
def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al.) on L2-normalised z."""
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(z), device=z.device)
    exp = torch.exp(sim) * (1 - eye)
    pos = (y[:, None] == y[None, :]).float() * (1 - eye)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    return -(log_prob * pos).sum(dim=1)[valid].div(n_pos[valid]).mean()
