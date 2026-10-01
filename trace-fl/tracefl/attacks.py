"""Byzantine attacks on client updates.

Data-level attacks (applied during local training, see simulation.py):
  lflip  label flipping y -> 9 - y
  scale  label flipping, then the update is multiplied by 10 (model replacement)

Update-level attacks (applied here; attackers are omniscient and collude):
  sign    -4 * own update
  gauss   large isotropic Gaussian noise
  alie    A Little Is Enough (Baruch et al., NeurIPS 2019): mu - z * sigma
  ipm     Inner-Product Manipulation (Xie et al., UAI 2019): -0.5 * mu
  aalie   adaptive ALIE: base + calibrated fresh noise (evades collusion tests)
  aipm    adaptive IPM
  aalief  adaptive ALIE with a FIXED noise direction per attacker (fakes persistence)
  aipmf   adaptive IPM with a fixed noise direction per attacker
"""
from math import erf, sqrt

import numpy as np

ATTACKS = ["none", "lflip", "sign", "gauss", "alie", "ipm", "scale",
           "aalie", "aipm", "aalief", "aipmf"]
DATA_ATTACKS = ("lflip", "scale")


def alie_z(n, nb):
    """ALIE z: largest z with Phi(z) < (n - s) / n, s = floor(n/2) + 1 - nb."""
    s = n // 2 + 1 - nb
    target = (n - s) / n
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return lo


def apply_attack(att, U, byz, rng, state=None):
    """Overwrite the Byzantine rows of U (n x d) in place and return U.

    `state` is a per-run dict; persistence-faking attacks keep their fixed noise there.
    """
    if state is None:
        state = {}
    benign = U[~byz]
    mu, sd = benign.mean(0), benign.std(0)
    nb = int(byz.sum())
    n = len(U)
    if att == "sign":
        U[byz] = -4.0 * U[byz]
    elif att == "gauss":
        U[byz] = rng.normal(0, 1.0, U[byz].shape).astype(np.float32) * np.linalg.norm(mu) / np.sqrt(U.shape[1]) * 30
    elif att == "alie":
        U[byz] = mu - alie_z(n, nb) * sd
    elif att == "ipm":
        U[byz] = -0.5 * mu
    elif att in ("aalie", "aipm", "aalief", "aipmf"):
        # Defence-aware: base malicious vector + independent noise, scaled so that the
        # pairwise cosine among attackers equals the median honest nearest-neighbour
        # cosine, then rescaled to the median honest norm (evades clipping).
        base = mu - alie_z(n, nb) * sd if att in ("aalie", "aalief") else -0.5 * mu
        Bn = benign / (np.linalg.norm(benign, axis=1, keepdims=True) + 1e-12)
        S = Bn @ Bn.T
        np.fill_diagonal(S, -1)
        tgt = float(np.clip(np.median(S.max(1)), 0.05, 0.99))
        b2 = float(base @ base)
        s2 = float((sd ** 2).sum()) + 1e-12
        eta = np.sqrt(max(b2 * (1 / tgt - 1), 0) / s2)
        if att.endswith("f"):  # one fixed direction per attacker for the whole run
            key = (nb, U.shape[1])
            if state.get("key") != key:
                state["key"] = key
                state["xi"] = np.random.default_rng(12345).standard_normal(key).astype(np.float32)
            xi = state["xi"]
        else:
            xi = rng.standard_normal((nb, U.shape[1])).astype(np.float32)
        V = base + eta * sd * xi
        hn = np.median(np.linalg.norm(benign, axis=1))
        U[byz] = V * (hn / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12))
    elif att == "scale":
        U[byz] = 10.0 * U[byz]
    elif att not in ("none", "lflip"):
        raise ValueError(f"unknown attack {att!r}; choose from {ATTACKS}")
    return U
