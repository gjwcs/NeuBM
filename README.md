# NeuBM: Mitigating Model Bias in Graph Neural Networks Through Neutral Input Calibration

Jiawei Gu, Ziyue Qiao, Xiao Luo · IJCAI 2025 · [Paper](https://www.ijcai.org/proceedings/2025/315) · [arXiv](https://arxiv.org/abs/2505.15180)

Graph neural networks trained on class-imbalanced data learn a bias toward the majority classes.
NeuBM (Neutral Bias Mitigation) measures this bias by feeding the trained model a *neutral graph*,
built only from training-set statistics (average size, average edge density, and the mean and
covariance of node features), and subtracts the resulting neutral logits from the logits of every
input: `ŷ = softmax(f_θ(G) − f_θ(G_neutral))`. The correction is post-hoc: the GNN is neither
retrained nor modified, and the neutral logits are computed once. On eight node-classification
benchmarks NeuBM gives the best F1-macro among the compared methods, with the largest gains on the
smallest classes.

This repository provides a concise implementation of the method. The detailed version with the full experimental code will be released later.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python demo.py
```

The demo builds a stochastic block model with skewed class sizes, trains a two-layer GCN on CPU with
plain cross-entropy, and prints per-class recall / F1 and macro-F1 on the test nodes before and after
NeuBM.

## Usage with any trained GNN

```python
from neubm import NeuBM

# model: a trained GNN called as model(x, edge_index) -> logits
#   node classification: logits of shape [num_nodes, C]; graph classification: [C] or [1, C]
neubm = NeuBM(model)                              # lam = 1.0: plain subtraction (Eq. 8)
neubm.fit(train_graphs)                           # list of (x, edge_index); for one graph: [(x, edge_index)]
proba = neubm.predict_proba(x, edge_index)        # softmax(f(G) - f(G_neutral))      (Eq. 9)
pred = proba.argmax(-1)

# other forward signatures
neubm = NeuBM(model, forward=lambda x, ei: model(x, ei, batch=None)).fit(train_graphs)
```

## API

| Object | Description |
|---|---|
| `graph_statistics(graphs)` | training-set statistics `GraphStats(n_bar, d_bar, mu, sigma)` (Eqs. 2–4) |
| `neutral_graph(stats, covariance='full', generator=None)` | samples `(x, edge_index)` with ⌊n̄⌋ nodes, edge probability d̄ and features ~ N(μ, Σ) (Eq. 5) |
| `NeuBM(model, lam=1.0, covariance='full', forward=None)` | post-hoc calibrator around a trained GNN |
| `NeuBM.fit(train_graphs, generator=None)` | builds the neutral graph and stores `neutral_logits` (Eq. 6) |
| `NeuBM.calibrate(logits)` | `logits - lam * neutral_logits` (Eqs. 8, 10 for `lam = 1`) |
| `NeuBM.logits / predict_proba / predict(x, edge_index)` | corrected logits, `softmax` probabilities (Eq. 9), labels |
| `GCN(in_dim, hidden, num_classes)` | two-layer GCN with a dense normalized adjacency, used by the demo |

`lam` weights the neutral logits; `lam = 1` (default) is NeuBM. Smaller values under-correct and
larger values over-correct the bias. For node-level models, the per-node neutral logits are averaged
into one reference vector that is subtracted from every node's logits.

## Citation

```bibtex
@inproceedings{gu2025neubm,
  title     = {NeuBM: Mitigating Model Bias in Graph Neural Networks Through Neutral Input Calibration},
  author    = {Gu, Jiawei and Qiao, Ziyue and Luo, Xiao},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25},
  pages     = {2829--2837},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/315}
}
```

## Contact

gjwcs@outlook.com
