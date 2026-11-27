"""NeuBM: Neutral Bias Mitigation for graph neural networks (Gu, Qiao, Luo, IJCAI 2025).

A trained GNN f_theta is evaluated twice: on the input graph G and on a *neutral graph* G_neutral
whose size, edge density and node-feature distribution follow the training-set statistics but carry
no class information. The neutral logits estimate the class bias the model learned from imbalanced
data and are subtracted from the input logits. The correction is post-hoc: the model is not retrained.

    L_neutral   = f_theta(G_neutral)                                  (Eq. 6)
    L           = f_theta(G)                                          (Eq. 7)
    L_corrected = L - L_neutral                                       (Eq. 8; B(L, L_neutral), Eq. 10)
    y_hat       = softmax(L_corrected)                                (Eq. 9)

Graphs are pairs (x, edge_index): node features x of shape [num_nodes, num_features] and an edge
list edge_index of shape [2, num_edges] (an undirected edge may be listed once or in both directions).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Graph = Tuple[torch.Tensor, torch.Tensor]

__all__ = ['GraphStats', 'graph_statistics', 'neutral_graph', 'NeuBM', 'GCN', 'normalized_adjacency']


# --------------------------------------------------------------------------- neutral graph (Sec. 2.2)
@dataclass
class GraphStats:
    """Training-set statistics used to build the neutral graph (Eqs. 2-4)."""
    n_bar: float            # average number of nodes
    d_bar: float            # average edge density 2|E| / (|V| (|V| - 1))
    mu: torch.Tensor        # mean node feature, shape [num_features]
    sigma: torch.Tensor     # node-feature covariance, shape [num_features, num_features]


def _num_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Number of distinct unordered pairs {u, v}, u != v, in an edge list."""
    u, v = edge_index[0].long(), edge_index[1].long()
    keep = u != v
    lo, hi = torch.minimum(u[keep], v[keep]), torch.maximum(u[keep], v[keep])
    return int(torch.unique(lo * num_nodes + hi).numel())


def graph_statistics(graphs: Iterable[Graph]) -> GraphStats:
    """Statistics of the training set D = {G_i = (V_i, E_i, X_i)}, i = 1..N.

        n_bar = 1/N sum_i |V_i|,   d_bar = 1/N sum_i 2|E_i| / (|V_i| (|V_i| - 1))         (Eq. 2)
        mu    = sum_i sum_{v in V_i} X_i[v] / sum_i |V_i|                                  (Eq. 3)
        Sigma = sum_i sum_{v in V_i} (X_i[v] - mu)(X_i[v] - mu)^T / sum_i |V_i|            (Eq. 4)

    No labels are used. For node classification on a single graph pass [(x, edge_index)] (N = 1).
    Moments are accumulated in float64 on the CPU.
    """
    sizes, densities, feats = [], [], []
    for x, edge_index in graphs:
        n = int(x.size(0))
        m = _num_undirected_edges(edge_index, n)
        sizes.append(n)
        densities.append(2.0 * m / (n * (n - 1)) if n > 1 else 0.0)
        feats.append(x.detach().to('cpu', torch.float64))
    if not sizes:
        raise ValueError('graph_statistics needs at least one training graph')
    X = torch.cat(feats)
    mu = X.mean(0)
    Xc = X - mu
    sigma = Xc.T @ Xc / X.size(0)
    return GraphStats(n_bar=sum(sizes) / len(sizes), d_bar=sum(densities) / len(densities), mu=mu, sigma=sigma)


def neutral_graph(stats: GraphStats, covariance: str = 'full', generator: Optional[torch.Generator] = None,
                  jitter: float = 1e-6, block: int = 1024) -> Graph:
    """Sample the neutral graph G_neutral = (V_neutral, E_neutral, X_neutral) (Sec. 2.2, Alg. 1).

    * |V_neutral| = floor(n_bar);
    * every distinct pair of nodes is joined by an undirected edge with probability d_bar;
    * X_neutral[v] ~ N(mu, Sigma) for every node v                                         (Eq. 5).

    covariance='diag' samples from N(mu, diag(Sigma)), a cheaper option for very high-dimensional
    features; 'full' (default) follows Eq. 5. A small jitter (relative to the mean variance) is added
    to Sigma so that the Cholesky factor exists for rank-deficient features.
    Returns (x [n, num_features] float32, edge_index [2, 2|E|] with both directions).
    """
    n = int(math.floor(stats.n_bar))
    # edges: independent Bernoulli(d_bar) draws on the upper triangle, in row blocks to bound memory
    us, vs = [], []
    for start in range(0, n, block):
        stop = min(start + block, n)
        hit = torch.rand(stop - start, n, generator=generator) < stats.d_bar
        i, j = hit.nonzero(as_tuple=True)
        i = i + start
        keep = j > i
        us.append(i[keep])
        vs.append(j[keep])
    u = torch.cat(us) if us else torch.zeros(0, dtype=torch.long)
    v = torch.cat(vs) if vs else torch.zeros(0, dtype=torch.long)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
    # node features (Eq. 5)
    mu, sigma = stats.mu.double(), stats.sigma.double()
    z = torch.randn(n, mu.numel(), generator=generator, dtype=torch.float64)
    if covariance == 'full':
        eps = jitter * max(float(sigma.diagonal().mean()), 1e-12)
        chol = torch.linalg.cholesky(sigma + eps * torch.eye(mu.numel(), dtype=torch.float64))
        x = mu + z @ chol.T
    elif covariance == 'diag':
        x = mu + z * sigma.diagonal().clamp_min(0).sqrt()
    else:
        raise ValueError("covariance must be 'full' or 'diag'")
    return x.float(), edge_index


# --------------------------------------------------------------------------- calibration (Sec. 2.3)
def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


class NeuBM:
    """Post-hoc Neutral Bias Mitigation around a trained GNN (Alg. 1).

    Args:
        model: the trained GNN, called as model(x, edge_index) -> logits. Node-level models return
            [num_nodes, C]; graph-level models return [C] or [1, C]. Pass `forward` for other signatures.
        lam: weight of the neutral logits in the calibration B(L, L_neutral) = L - lam * L_neutral.
            lam = 1 (default) is NeuBM's plain subtraction (Eqs. 8, 10). The paper's scaling study
            (Sec. 3.4, Fig. 5) reports that lam < 1 under-corrects and lam > 1 over-corrects the bias;
            weighting the correction term, rather than rescaling the whole difference L - L_neutral
            (which would leave the argmax unchanged), gives that behaviour.
        covariance: 'full' (Eq. 5) or 'diag' feature sampling for the neutral graph.
        forward: optional callable (x, edge_index) -> logits used instead of model(x, edge_index).

    For node-level models, f_theta(G_neutral) gives one logit vector per neutral node; these are
    averaged into a single reference L_neutral in R^C that is subtracted from the logits of every node.
    """

    def __init__(self, model: nn.Module, lam: float = 1.0, covariance: str = 'full',
                 forward: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        self.model = model
        self.lam = float(lam)
        self.covariance = covariance
        self.forward = forward if forward is not None else model
        self.stats: Optional[GraphStats] = None
        self.neutral: Optional[Graph] = None
        self.neutral_logits: Optional[torch.Tensor] = None

    def _run(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            return self.forward(x, edge_index)
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def fit(self, train_graphs: Iterable[Graph], generator: Optional[torch.Generator] = None) -> 'NeuBM':
        """Build G_neutral from training-set statistics and compute L_neutral (Eqs. 2-6, Alg. 1)."""
        self.stats = graph_statistics(train_graphs)
        x_n, edge_index_n = neutral_graph(self.stats, covariance=self.covariance, generator=generator)
        device = _device_of(self.model)
        self.neutral = (x_n.to(device), edge_index_n.to(device))
        out = self._run(*self.neutral)                                   # Eq. 6
        self.neutral_logits = out.reshape(-1, out.size(-1)).mean(0)      # [C]
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """B(L, L_neutral) = L - lam * L_neutral; lam = 1 gives Eqs. 8 and 10."""
        if self.neutral_logits is None:
            raise RuntimeError('call fit() before calibrate()')
        return logits - self.lam * self.neutral_logits.to(device=logits.device, dtype=logits.dtype)

    @torch.no_grad()
    def logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Corrected logits L - lam * L_neutral with L = f_theta(G) (Eqs. 7-8)."""
        return self.calibrate(self._run(x, edge_index))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calibrated class probabilities softmax(L_corrected) (Eq. 9)."""
        return F.softmax(self.logits(x, edge_index), dim=-1)

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calibrated class predictions argmax softmax(L_corrected)."""
        return self.predict_proba(x, edge_index).argmax(-1)


# --------------------------------------------------------------------------- a tiny GNN for the demo
def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Dense GCN propagation matrix D^-1/2 (A + I) D^-1/2 of an undirected graph."""
    a = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    if edge_index.numel():
        a[edge_index[0], edge_index[1]] = 1.0
    a = ((a + a.T) > 0).float()
    a.fill_diagonal_(1.0)
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


class GCN(nn.Module):
    """Two-layer GCN with a dense normalized adjacency: forward(x, edge_index) -> logits [num_nodes, C]."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        a = normalized_adjacency(edge_index, x.size(0))
        h = F.relu(a @ self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return a @ self.lin2(h)
