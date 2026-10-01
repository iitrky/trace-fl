"""Server-side aggregation rules.

Every aggregator has the signature  agg(U, state, **kw) -> g
  U      (n, d) float32 matrix of client updates for this round
  state  dict persisted across rounds for this run (reputations, histories, ...)
  kw     f (number of Byzantine clients, used only by rules that need it),
         root_update (callable, FLTrust only), plus rule-specific options.

Proposed:
  tracefl       TRACE-FL core (Steps 1-4 + reputation-weighted aggregation)
  tracefl_plus  TRACE-FL+ (adds Step 5, history-aware collusion weight)
Baselines:
  fedavg, krum, median, trmean, cclip, fltrust, bulyan, foolsgold, rfa
"""
import numpy as np


# ------------------------------------------------------------------ baselines
def fedavg(U, st, **k):
    return U.mean(0)


def median(U, st, **k):
    return np.median(U, 0)


def trmean(U, st, f, **k):
    S = np.sort(U, 0)
    return S[f:len(U) - f].mean(0)


def krum(U, st, f, **k):
    n = len(U)
    D = ((U[:, None] - U[None]) ** 2).sum(-1)
    sc = [np.sort(D[i])[1:n - f - 1].sum() for i in range(n)]
    return U[int(np.argmin(sc))]


def cclip(U, st, tau=None, iters=3, **k):
    """Centered Clipping (Karimireddy et al., ICML 2021)."""
    v = st.get("prev", np.zeros(U.shape[1], np.float32))
    tau = tau or np.median(np.linalg.norm(U - v, axis=1))
    for _ in range(iters):
        d = U - v
        nd = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        v = v + (d * np.minimum(1, tau / nd)).mean(0)
    st["prev"] = v
    return v


def fltrust(U, st, root_update, **k):
    """FLTrust (Cao et al., NDSS 2021); needs a clean server root dataset."""
    g0 = root_update()
    n0 = np.linalg.norm(g0) + 1e-12
    nu = np.linalg.norm(U, axis=1) + 1e-12
    ts = np.maximum(U @ g0 / (nu * n0), 0)
    if ts.sum() == 0:
        return np.zeros_like(g0)
    return (ts[:, None] * U * (n0 / nu)[:, None]).sum(0) / ts.sum()


def bulyan(U, st, f, **k):
    """Bulyan (El Mhamdi et al., ICML 2018). Returns None when n < 4f + 3."""
    n = len(U)
    if n < 4 * f + 3:
        return None
    D = ((U[:, None] - U[None]) ** 2).sum(-1)
    rem, sel = list(range(n)), []
    theta = n - 2 * f
    while len(sel) < theta:
        kk = max(len(rem) - f - 2, 1)
        sc = [np.sort(D[i, rem])[1:kk + 1].sum() for i in rem]
        j = rem[int(np.argmin(sc))]
        sel.append(j)
        rem.remove(j)
    S = U[sel]
    beta = theta - 2 * f
    idx = np.argsort(np.abs(S - np.median(S, 0)), 0)[:beta]
    return np.take_along_axis(S, idx, 0).mean(0)


def foolsgold(U, st, **k):
    """FoolsGold (Fung et al., RAID 2020) on raw cumulative update histories."""
    n = len(U)
    H = st.get("hist", np.zeros_like(U)) + U
    st["hist"] = H
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    cs = Hn @ Hn.T
    np.fill_diagonal(cs, -np.inf)
    v = cs.max(1)
    for i in range(n):
        for j in range(n):
            if i != j and v[j] > v[i]:
                cs[i, j] *= v[i] / v[j]
    a = np.clip(1 - cs.max(1), 0, 1)
    a = a / (a.max() + 1e-12)
    a[a >= 1] = 0.99
    a = np.clip(np.log(a / (1 - a) + 1e-12) + 0.5, 0, 1)
    if a.sum() < 1e-9:
        return np.median(U, 0)
    return (a[:, None] * U).sum(0) / a.sum()


def rfa(U, st, T=3, nu=1e-6, **k):
    """Geometric median by smoothed Weiszfeld (Pillutla et al., IEEE TSP 2022)."""
    z = U.mean(0)
    for _ in range(T):
        b = 1.0 / np.maximum(nu, np.linalg.norm(U - z, axis=1))
        z = (b[:, None] * U).sum(0) / b.sum()
    return z


# ------------------------------------------------------------------ TRACE-FL
def _robust_z(x, eps):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)


def tracefl(U, st, gamma=0.8, clip=True, rep=True, dist=True, coll=True, dup=0.999, pers=True, **k):
    """TRACE-FL core aggregation (Steps 1-4 of the paper).

    Step 1  clip every update to the median norm
    Step 2  reference m = coordinate-wise median of clipped updates
    Step 3  q_i = c_i * delta_i * kappa_i * pi_i
              c_i      direction consistency   max(0, cos(u_i, m))
              delta_i  robust distance         exp(-max(0, z_d - 1))
              kappa_i  collusion (NN cosine)   exp(-2 max(0, z_s - 2)) * [s_i < dup]
              pi_i     trajectory persistence  max(0, cos(dev_i^t, dev_i^{t-1}))^2
    Step 4  r_i <- gamma r_i + (1 - gamma) q_i   (r_i = q_i in the first round)
    Output  reputation-weighted mean of clipped updates.
    The boolean flags switch individual components off for ablations.
    """
    norms = np.linalg.norm(U, axis=1) + 1e-12
    tau = np.median(norms)
    Uc = U * np.minimum(1, tau / norms)[:, None] if clip else U
    ref = np.median(Uc, 0)
    nr = np.linalg.norm(ref) + 1e-12
    q = np.maximum((Uc @ ref) / (np.linalg.norm(Uc, axis=1) * nr + 1e-12), 0)
    if dist:
        z = _robust_z(np.linalg.norm(Uc - ref, axis=1), 1e-12)
        q = q * np.exp(-np.maximum(z - 1.0, 0))
    if coll:
        Un = Uc / (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-12)
        S = Un @ Un.T
        np.fill_diagonal(S, -1)
        nn = S.max(1)
        zc = _robust_z(nn, 1e-6)
        q = q * np.exp(-2.0 * np.maximum(zc - 2.0, 0)) * (nn < dup)
    D = Uc - ref
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    if pers and "pd" in st:
        p = (D * st["pd"]).sum(1)
        q = q * np.maximum(p, 0) ** 2
    st["pd"] = D
    r = st.get("rep", q)
    r = gamma * r + (1 - gamma) * q if rep else q
    st["rep"] = r
    if r.sum() < 1e-9:
        return ref
    return (r[:, None] * Uc).sum(0) / r.sum()


def foolsgold_weights(H):
    """FoolsGold weights a_i in [0, 1] from cumulative histories H (n x d), with pardoning."""
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    cs = Hn @ Hn.T
    np.fill_diagonal(cs, -np.inf)
    v = cs.max(1)
    vv = np.maximum(v, 1e-12)
    ratio = np.minimum(1.0, vv[:, None] / vv[None, :])
    cs = np.where(np.isfinite(cs), cs * ratio, cs)
    a = np.clip(1 - cs.max(1), 0, 1)
    a = a / (a.max() + 1e-12)
    a[a >= 1] = 0.99
    return np.clip(np.log(a / (1 - a) + 1e-12) + 0.5, 0, 1)


def tracefl_plus(U, st, **k):
    """TRACE-FL+ (proposed): TRACE-FL reputation x FoolsGold-style weight computed on
    cumulative histories of CLIPPED updates (Step 5), then weighted aggregation (Step 6)."""
    norms = np.linalg.norm(U, axis=1) + 1e-12
    Uc = U * np.minimum(1, np.median(norms) / norms)[:, None]
    H = st.get("chist", np.zeros_like(U)) + Uc
    st["chist"] = H
    g_core = tracefl(U, st, **k)  # updates st["rep"]
    w = st["rep"] * foolsgold_weights(H)
    st["weights"] = w
    if w.sum() < 1e-9:
        return g_core
    return (w[:, None] * Uc).sum(0) / w.sum()


AGGREGATORS = {
    "fedavg": fedavg, "krum": krum, "median": median, "trmean": trmean, "cclip": cclip,
    "fltrust": fltrust, "bulyan": bulyan, "foolsgold": foolsgold, "rfa": rfa,
    "tracefl": tracefl, "tracefl_plus": tracefl_plus,
}
# names used in the paper's tables / earlier scripts
ALIASES = {"trace": "tracefl", "trace-fl": "tracefl", "trace3": "tracefl_plus",
           "trace-fl+": "tracefl_plus", "tracefl+": "tracefl_plus"}


def get_aggregator(name):
    key = name.lower()
    key = ALIASES.get(key, key)
    if key not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(AGGREGATORS)}")
    return AGGREGATORS[key]
