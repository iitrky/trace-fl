"""Unsupervised single-layer k-means convolutional features for CIFAR-10
(Coates, Lee & Ng, AISTATS 2011): 6x6 patches, ZCA whitening, 256 centroids,
triangle activation, stride 2, four-quadrant sum pooling -> 1,024 dims.
The extractor never sees labels and is frozen before federated training."""
import time

import numpy as np


def _norm(p):
    return (p - p.mean(1, keepdims=True)) / np.sqrt(p.var(1, keepdims=True) + 0.01)


def build_cifar_features(npz_path, out_path, P=6, K=256, n_patches=100000, stride=2, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    d = np.load(npz_path)
    Xtr = d["train_x"].astype(np.float32) / 255.
    Xte = d["test_x"].astype(np.float32) / 255.
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(Xtr), n_patches)
    r = rng.integers(0, 32 - P + 1, n_patches)
    c = rng.integers(0, 32 - P + 1, n_patches)
    pat = _norm(np.stack([Xtr[i, a:a + P, b:b + P].ravel() for i, a, b in zip(ii, r, c)]))
    M = pat.mean(0)
    e, V = np.linalg.eigh(np.cov(pat - M, rowvar=False))
    Wz = V @ np.diag(1 / np.sqrt(e + 0.1)) @ V.T
    km = MiniBatchKMeans(K, batch_size=4096, n_init=1, max_iter=30, random_state=0).fit((pat - M) @ Wz)
    Cn = km.cluster_centers_.astype(np.float32)

    def extract(X):
        out = np.zeros((len(X), 4 * K), np.float32)
        pos = range(0, 32 - P + 1, stride)
        L = len(pos)
        for s in range(0, len(X), 500):
            B = X[s:s + 500]
            nb = len(B)
            pp = np.stack([B[:, a:a + P, b:b + P].reshape(nb, -1) for a in pos for b in pos], 1)
            pp = ((_norm(pp.reshape(-1, P * P * 3)) - M) @ Wz).astype(np.float32)
            z = np.sqrt(np.maximum((pp ** 2).sum(1, keepdims=True) - 2 * pp @ Cn.T + (Cn ** 2).sum(1), 0))
            f = np.maximum(0, z.mean(1, keepdims=True) - z).reshape(nb, L, L, K)
            h = L // 2
            out[s:s + nb] = np.concatenate([f[:, :h, :h].sum((1, 2)), f[:, :h, h:].sum((1, 2)),
                                            f[:, h:, :h].sum((1, 2)), f[:, h:, h:].sum((1, 2))], 1)
        return out

    t = time.time()
    Ftr, Fte = extract(Xtr), extract(Xte)
    print(f"CIFAR-10 feature extraction: {time.time() - t:.0f}s")
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-3
    np.savez(out_path, Xtr=(Ftr - m) / s, ytr=d["train_y"], Xte=(Fte - m) / s, yte=d["test_y"])
