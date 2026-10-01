"""End-to-end CNN federated training on raw CIFAR-10 images (JAX; run on a GPU).

The paper's CIFAR-10 results use frozen features to keep the full attack grid
feasible on one CPU core. This script trains a small CNN end to end with the same
aggregators and attacks, for the GPU-scale version of the experiments.

Setup:
  pip install -U "jax[cuda12]"          # GPU build of JAX (see jax.readthedocs.io)
  python cnn/cnn_cifar10.py --seeds 0 1 2 3 4 --rounds 100                  # full grid
  python cnn/cnn_cifar10.py --methods tracefl_plus cclip --attacks aipm --byz 0.4

Results are appended to results/cnn_runs.json (resumable) and summarised at the end.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from tracefl.aggregators import get_aggregator  # noqa: E402
from tracefl.attacks import DATA_ATTACKS, apply_attack  # noqa: E402
from tracefl.data import cifar10_npz, dirichlet_split  # noqa: E402

_d = np.load(cifar10_npz())
Xtr = _d["train_x"].astype(np.float32) / 255.
Xte = _d["test_x"].astype(np.float32) / 255.
_mu, _sd = Xtr.mean((0, 1, 2)), Xtr.std((0, 1, 2))
Xtr, Xte = (Xtr - _mu) / _sd, (Xte - _mu) / _sd
ytr, yte = _d["train_y"], _d["test_y"]


def init(key):
    k = jax.random.split(key, 4)
    he = lambda kk, s, fan: jax.random.normal(kk, s) * jnp.sqrt(2 / fan)  # noqa: E731
    return dict(c1=he(k[0], (3, 3, 3, 32), 27), b1=jnp.zeros(32),
                c2=he(k[1], (3, 3, 32, 64), 288), b2=jnp.zeros(64),
                w3=he(k[2], (8 * 8 * 64, 128), 4096), b3=jnp.zeros(128),
                w4=jax.random.normal(k[3], (128, 10)) * jnp.sqrt(1 / 128), b4=jnp.zeros(10))


def forward(p, x):
    dn = ("NHWC", "HWIO", "NHWC")
    h = jax.nn.relu(jax.lax.conv_general_dilated(x, p["c1"], (1, 1), "SAME", dimension_numbers=dn) + p["b1"])
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    h = jax.nn.relu(jax.lax.conv_general_dilated(h, p["c2"], (1, 1), "SAME", dimension_numbers=dn) + p["b2"])
    h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    h = jax.nn.relu(h.reshape(h.shape[0], -1) @ p["w3"] + p["b3"])
    return h @ p["w4"] + p["b4"]


_, unravel = ravel_pytree(init(jax.random.PRNGKey(0)))
LR = 0.05


def _loss(w, x, y):
    lg = forward(unravel(w), x)
    return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(len(y)), y])


_grad = jax.grad(_loss)


@jax.jit
def local(w, xb, yb):  # xb: (steps, B, 32, 32, 3)
    def step(v, b):
        return v - LR * _grad(v, b[0], b[1]), None
    v, _ = jax.lax.scan(step, w, (xb, yb))
    return v - w


_pred = jax.jit(lambda w, x: forward(unravel(w), x).argmax(1))


def evaluate(w):
    pr = np.concatenate([np.asarray(_pred(w, Xte[i:i + 2000])) for i in range(0, len(Xte), 2000)])
    return float((pr == yte).mean())


def run(agg_name, att, n_clients=40, frac_byz=0.2, rounds=100, steps=20, B=32, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    agg = get_aggregator(agg_name)
    w = jnp.asarray(ravel_pytree(init(jax.random.PRNGKey(seed)))[0])
    parts = dirichlet_split(ytr, n_clients, alpha, rng)
    nb = int(frac_byz * n_clients)
    byz = np.zeros(n_clients, bool)
    byz[rng.choice(n_clients, nb, replace=False)] = True
    root = rng.choice(len(ytr), 100, replace=False)
    st, ast, accs = {}, {}, []
    for t in range(rounds):
        U = []
        for k in range(n_clients):
            src = parts[k] if len(parts[k]) else np.arange(len(ytr))
            idx = rng.choice(src, (steps, B))
            yb = ytr[idx].copy()
            if byz[k] and att in DATA_ATTACKS:
                yb = 9 - yb
            U.append(np.asarray(local(w, jnp.asarray(Xtr[idx]), jnp.asarray(yb)), np.float32))
        U = np.stack(U)
        if nb and att not in ("none", "lflip"):
            U = apply_attack(att, U, byz, rng, ast)

        def root_update():
            ri = rng.choice(root, (steps, B))
            return np.asarray(local(w, jnp.asarray(Xtr[ri]), jnp.asarray(ytr[ri])), np.float32)

        g = agg(U, st, f=nb, root_update=root_update)
        if g is None:
            return None
        w = w + jnp.asarray(np.nan_to_num(g))
        if t >= rounds - 5:
            accs.append(evaluate(w))
    return {"acc": float(np.mean(accs))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=["fedavg", "median", "trmean", "cclip", "fltrust", "bulyan",
                                                     "foolsgold", "rfa", "tracefl", "tracefl_plus"])
    ap.add_argument("--attacks", nargs="+", default=None, help="override the attack grid")
    ap.add_argument("--byz", type=float, nargs="+", default=None, help="override Byzantine fractions")
    a = ap.parse_args()
    if a.attacks or a.byz:
        grid = [(fb, t) for fb in (a.byz or [0.2]) for t in (a.attacks or ["none"])]
    else:
        grid = [(0.2, t) for t in ["none", "lflip", "sign", "gauss", "alie", "ipm", "scale", "aalie", "aipm"]] + \
               [(0.4, t) for t in ["sign", "ipm", "aalie", "aipm", "aalief", "aipmf"]]
    fn = os.path.join(ROOT, "results", "cnn_runs.json")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for seed in a.seeds:
        for fb, att in grid:
            for m in a.methods:
                k = f"{seed}|{fb}|{m}|{att}|{a.rounds}"
                if k in res:
                    continue
                res[k] = run(m, att, frac_byz=fb, rounds=a.rounds, seed=seed)
                json.dump(res, open(fn, "w"), indent=1)
                print(k, "n/a" if res[k] is None else round(100 * res[k]["acc"], 2), flush=True)
    summ = {}
    for k, v in res.items():
        s, fb, m, att, R = k.split("|")
        if v is not None and int(R) == a.rounds:
            summ.setdefault((fb, att, m), []).append(100 * v["acc"])
    print("\nByz  attack   method         accuracy (mean +- std over seeds)")
    for (fb, att, m), v in sorted(summ.items()):
        print(f"{fb:>4} {att:<8} {m:<14} {np.mean(v):6.2f} +- {np.std(v):.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
