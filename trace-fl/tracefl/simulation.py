"""Federated training simulation (synchronous FedAvg-style rounds, full participation)."""
import numpy as np

from .aggregators import get_aggregator
from .attacks import DATA_ATTACKS, apply_attack
from .data import dirichlet_split, load_cifar_features, load_digits_split, load_fmnist
from .models import MLP, local_update

# dataset -> (hidden units, learning rate, batch size, local epochs), as in the paper
HPARAMS = {"fmnist": (64, 0.05, 32, 1), "cifar": (128, 0.02, 32, 1), "digits": (32, 0.05, 16, 2)}
_CACHE = {}


def load_dataset(name, seed=0):
    """Return (Xtr, ytr, Xte, yte). Fashion-MNIST and CIFAR features are cached in memory."""
    if name == "fmnist":
        if "fmnist" not in _CACHE:
            _CACHE["fmnist"] = load_fmnist()
        return _CACHE["fmnist"]
    if name == "cifar":
        if "cifar" not in _CACHE:
            _CACHE["cifar"] = load_cifar_features()
        return _CACHE["cifar"]
    if name == "digits":
        return load_digits_split(seed)
    raise ValueError(f"unknown dataset {name!r}; choose fmnist, digits or cifar")


def run_fl(dataset, aggregator, attack="none", alpha=0.5, n_clients=40, frac_byz=0.2,
           rounds=30, seed=0, agg_kwargs=None, eval_last=5, return_history=False):
    """Run one federated training job and return a result dict.

    acc      mean test accuracy over the last `eval_last` rounds (paper metric)
    rep_byz  mean final weight of Byzantine clients (reputation-based rules only)
    rep_ben  mean final weight of honest clients
    Returns None for Bulyan when it is not applicable (n < 4f + 3).
    """
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = load_dataset(dataset, seed)
    hid, lr, bs, ep = HPARAMS[dataset]
    agg = get_aggregator(aggregator)
    model = MLP(Xtr.shape[1], hid, 10)
    w = model.init(rng)
    parts = dirichlet_split(ytr, n_clients, alpha, rng)
    nbyz = int(frac_byz * n_clients)
    byz = np.zeros(n_clients, bool)
    byz[rng.choice(n_clients, nbyz, replace=False)] = True
    root = rng.choice(len(ytr), 100, replace=False)  # FLTrust's clean root set
    state, attack_state, accs = {}, {}, []

    for t in range(rounds):
        U = []
        for k in range(n_clients):
            X, y = Xtr[parts[k]], ytr[parts[k]]
            if len(y) == 0:
                U.append(np.zeros(model.n, np.float32))
                continue
            if byz[k] and attack in DATA_ATTACKS:
                y = 9 - y
            U.append(local_update(model, w, X, y, rng, lr, ep, bs))
        U = np.array(U, np.float32)
        if nbyz and attack not in ("none", "lflip"):
            U = apply_attack(attack, U, byz, rng, attack_state)

        def root_update():
            return local_update(model, w, Xtr[root], ytr[root], rng, lr, ep, bs)

        g = agg(U, state, f=nbyz, root_update=root_update, **(agg_kwargs or {}))
        if g is None:
            return None
        w = w + g
        if not np.all(np.isfinite(w)):
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if t >= rounds - eval_last or return_history:
            accs.append(model.accuracy(w, Xte, yte))

    out = {"acc": float(np.mean(accs[-eval_last:]))}
    weights = state.get("weights", state.get("rep"))
    if weights is not None:
        out["rep_byz"] = float(weights[byz].mean()) if nbyz else 0.0
        out["rep_ben"] = float(weights[~byz].mean())
    if return_history:
        out["history"] = accs
    return out
