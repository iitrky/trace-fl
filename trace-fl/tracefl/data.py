"""Datasets, downloads and the Dirichlet non-IID split.

Fashion-MNIST: official zalandoresearch GitHub repository.
UCI Digits:    bundled with scikit-learn.
CIFAR-10:      GitHub image mirror (YoongiKim/CIFAR-10-images) -> cifar10.npz,
               then frozen unsupervised k-means conv features (Coates et al., 2011).

Set the environment variable TRACEFL_DATA to change the data directory.
"""
import gzip
import io
import os
import urllib.request
import zipfile

import numpy as np

DATA_DIR = os.environ.get(
    "TRACEFL_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
FMNIST_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{}.gz"
FMNIST_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
CIFAR_ZIP_URL = "https://codeload.github.com/YoongiKim/CIFAR-10-images/zip/refs/heads/master"


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


# ------------------------------------------------------------------ Fashion-MNIST
def load_fmnist(n_train=20000, data_dir=DATA_DIR):
    """Standardised Fashion-MNIST. Statistics use the full 60k training set;
    the first `n_train` training images are returned (paper: 20,000)."""
    d = os.path.join(data_dir, "fashion-mnist")
    for f in FMNIST_FILES:
        _download(FMNIST_URL.format(f), os.path.join(d, f + ".gz"))

    def img(f):
        with gzip.open(os.path.join(d, f + ".gz")) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.

    def lab(f):
        with gzip.open(os.path.join(d, f + ".gz")) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)

    Xtr, ytr = img(FMNIST_FILES[0]), lab(FMNIST_FILES[1])
    Xte, yte = img(FMNIST_FILES[2]), lab(FMNIST_FILES[3])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xtr[:n_train], ytr[:n_train], Xte, yte


# ------------------------------------------------------------------ UCI Digits
def load_digits_split(seed):
    """UCI Digits (8x8 images), seeded 1,400 / 397 train/test split."""
    from sklearn.datasets import load_digits
    rng = np.random.default_rng(seed)
    dd = load_digits()
    X = dd.data.astype(np.float32) / 16.
    y = dd.target
    p = rng.permutation(len(y))
    X, y = X[p], y[p]
    mu, sd = X[:1400].mean(0), X[:1400].std(0) + 1e-3
    return (X[:1400] - mu) / sd, y[:1400], (X[1400:] - mu) / sd, y[1400:]


# ------------------------------------------------------------------ CIFAR-10
def cifar10_npz(data_dir=DATA_DIR):
    """Download the CIFAR-10 image mirror once and cache it as uint8 arrays."""
    out = os.path.join(data_dir, "cifar10.npz")
    if os.path.exists(out):
        return out
    from PIL import Image
    zpath = _download(CIFAR_ZIP_URL, os.path.join(data_dir, "cifar10_images.zip"))
    arrays = {}
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))]
        root = names[0].split("/")[0]
        classes = sorted({n.split("/")[2] for n in names if n.startswith(f"{root}/train/")})
        for split in ("train", "test"):
            X, Y = [], []
            for ci, c in enumerate(classes):
                for f in sorted(n for n in names if n.startswith(f"{root}/{split}/{c}/")):
                    X.append(np.asarray(Image.open(io.BytesIO(z.read(f))).convert("RGB"), np.uint8))
                    Y.append(ci)
            arrays[split + "_x"] = np.stack(X)
            arrays[split + "_y"] = np.array(Y, np.int64)
    np.savez(out, **arrays)
    return out


def load_cifar_features(data_dir=DATA_DIR):
    """Frozen k-means convolutional features (1,024-d). Built on first use
    (about 3-5 minutes on one CPU core) and cached in data/cifar_feats.npz."""
    path = os.path.join(data_dir, "cifar_feats.npz")
    if not os.path.exists(path):
        from .features import build_cifar_features
        build_cifar_features(cifar10_npz(data_dir), path)
    d = np.load(path)
    return d["Xtr"].astype(np.float32), d["ytr"], d["Xte"].astype(np.float32), d["yte"]


# ------------------------------------------------------------------ non-IID split
def dirichlet_split(y, n_clients, alpha, rng, n_classes=10):
    """Per-class Dirichlet(alpha) allocation of sample indices to clients."""
    idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        ic = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
        for k, part in enumerate(np.split(ic, cuts)):
            idx[k] += part.tolist()
    return [np.array(i) for i in idx]
