"""NeuBM demo on a synthetic class-imbalanced graph (CPU, a few seconds).

A stochastic block model with skewed class sizes is generated, a two-layer GCN is trained on an
equally skewed set of labelled nodes with plain cross-entropy, and the test nodes are classified
before and after Neutral Bias Mitigation. Per-class recall / F1 and macro-F1 are computed here.

    python demo.py [--seed 0] [--epochs 200]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from neubm import GCN, NeuBM


def sbm_graph(sizes, p_in, p_out, num_features, signal, gen):
    """Stochastic block model with class sizes `sizes` and class-dependent Gaussian node features."""
    y = torch.cat([torch.full((s,), c, dtype=torch.long) for c, s in enumerate(sizes)])
    n = y.numel()
    same = (y[:, None] == y[None, :]).float()
    prob = same * (p_in - p_out) + p_out
    upper = torch.triu((torch.rand(n, n, generator=gen) < prob).float(), diagonal=1)
    u, v = upper.nonzero(as_tuple=True)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
    centers = signal * torch.randn(len(sizes), num_features, generator=gen)
    x = centers[y] + torch.randn(n, num_features, generator=gen)
    return x, edge_index, y


def stratified_split(y, train_frac, gen):
    """The same fraction of every class is labelled, so the training labels keep the class imbalance."""
    train = torch.zeros_like(y, dtype=torch.bool)
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        train[idx[:max(1, int(round(train_frac * idx.numel())))]] = True
    return train, ~train


def per_class_scores(y_true, y_pred, num_classes):
    """Per-class recall and F1 from the confusion counts; macro-F1 is their unweighted mean."""
    recall, f1 = np.zeros(num_classes), np.zeros(num_classes)
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recall[c] = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1[c] = 2 * precision * recall[c] / (precision + recall[c]) if precision + recall[c] else 0.0
    return recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()

    sizes = [400, 200, 100, 50, 25]                     # class 0 = majority, class 4 = smallest
    x, edge_index, y = sbm_graph(sizes, p_in=0.03, p_out=0.005, num_features=16, signal=0.6, gen=gen)
    train, test = stratified_split(y, train_frac=0.2, gen=gen)
    num_classes = len(sizes)

    model = GCN(x.size(1), 32, num_classes, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(args.epochs):                        # plain cross-entropy on imbalanced labels
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[train], y[train])
        loss.backward()
        opt.step()

    # NeuBM: statistics of the (single) training graph -> neutral graph -> neutral logits (Eqs. 2-6)
    neubm = NeuBM(model).fit([(x, edge_index)], generator=gen)
    model.eval()
    with torch.no_grad():
        pred_before = model(x, edge_index).argmax(-1)                # softmax(L), Eq. 7
    pred_after = neubm.predict(x, edge_index)                        # softmax(L - L_neutral), Eqs. 8-9

    yt = y[test].numpy()
    rec_b, f1_b = per_class_scores(yt, pred_before[test].numpy(), num_classes)
    rec_a, f1_a = per_class_scores(yt, pred_after[test].numpy(), num_classes)

    print('graph: %d nodes, %d edges, class sizes %s, %d labelled nodes'
          % (y.numel(), edge_index.size(1) // 2, sizes, int(train.sum())))
    print('neutral graph: %d nodes, edge density %.4f, L_neutral = %s'
          % (neubm.neutral[0].size(0), neubm.stats.d_bar,
             np.array2string(neubm.neutral_logits.numpy(), precision=2)))
    print()
    print('class  test nodes   recall GCN  recall NeuBM   F1 GCN  F1 NeuBM')
    for c in range(num_classes):
        print('%5d  %10d   %10.3f  %12.3f   %6.3f  %8.3f'
              % (c, int((yt == c).sum()), rec_b[c], rec_a[c], f1_b[c], f1_a[c]))
    print('macro-F1: GCN %.3f -> NeuBM %.3f' % (f1_b.mean(), f1_a.mean()))
    print('accuracy: GCN %.3f -> NeuBM %.3f'
          % ((pred_before[test].numpy() == yt).mean(), (pred_after[test].numpy() == yt).mean()))
    print('done in %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
