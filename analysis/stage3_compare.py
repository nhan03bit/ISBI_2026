"""Paired patient bootstrap for Phase A prediction exports (development only)."""
import argparse
import json
import numpy as np
from sklearn.metrics import average_precision_score


def compare(first, second, repeats=1000, seed=42, unseen_only=False):
    for key in ('keys', 'labels', 'names', 'patients', 'unseen', 'tail', 'head'):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f'Prediction alignment differs: {key}')
    mask = first['unseen'] if unseen_only else np.ones(len(first['labels']), dtype=bool)
    y, a, patients = [first[k][mask] for k in ('labels', 'scores', 'patients')]
    b = second['scores'][mask]
    if not len(y):
        raise ValueError('No images in requested population')
    support = np.flatnonzero(y.sum(0) > 0)
    groups = dict(overall=support, tail=np.intersect1d(first['tail'], support),
                  head=np.intersect1d(first['head'], support))
    _, patient_indices = np.unique(patients, return_inverse=True)
    n = patient_indices.max() + 1

    def aps(scores, weights):
        return np.array([average_precision_score(y[:, c], scores[:, c], sample_weight=weights)
                         if np.dot(y[:, c], weights) > 0 else np.nan for c in range(y.shape[1])])

    base_a, base_b = aps(a, np.ones(len(y))), aps(b, np.ones(len(y)))
    rng = np.random.default_rng(seed)
    differences = {k: [] for k in groups}
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(n, size=n), minlength=n)[patient_indices]
        delta = aps(b, multiplicity) - aps(a, multiplicity)
        for k, ids in groups.items():
            # Do not silently change class membership in bootstrap replicates.
            if len(ids) and np.isfinite(delta[ids]).all():
                differences[k].append(float(delta[ids].mean()))
    return dict(population='unseen patients (all projections)' if unseen_only else 'full validation',
                images=len(y), patients=int(n), positive_counts=y.sum(0).tolist(),
                per_class_delta=[float(v) if np.isfinite(v) else None for v in base_b - base_a],
                groups={k: dict(labels=first['names'][ids].tolist(),
                                delta=float((base_b[ids] - base_a[ids]).mean()) if len(ids) else None,
                                interval95=np.quantile(differences[k], [.025, .975]).tolist() if differences[k] else None,
                                valid_bootstraps=len(differences[k])) for k, ids in groups.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--unseen-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare(np.load(args.baseline), np.load(args.candidate), args.repeats,
                             unseen_only=args.unseen_only), indent=2))
