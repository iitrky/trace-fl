"""Fast unit tests (no downloads): python -m pytest -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracefl.aggregators import AGGREGATORS, foolsgold_weights, get_aggregator, tracefl  # noqa: E402
from tracefl.attacks import apply_attack  # noqa: E402
from tracefl.data import dirichlet_split  # noqa: E402
from tracefl.models import MLP, local_update  # noqa: E402


def honest_updates(n=30, d=200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=d)
    return (base + 0.8 * rng.normal(size=(n, d))).astype(np.float32)


@pytest.mark.parametrize("name", sorted(AGGREGATORS))
def test_aggregators_return_vector(name):
    U = honest_updates()
    g = get_aggregator(name)(U, {}, f=6, root_update=lambda: U.mean(0))
    assert g is not None and g.shape == (U.shape[1],) and np.all(np.isfinite(g))


def test_aliases():
    assert get_aggregator("TRACE3") is AGGREGATORS["tracefl_plus"]
    assert get_aggregator("TRACE-FL") is AGGREGATORS["tracefl"]


def test_lemma1_clipped_norm_bound():
    """Lemma 1: with f < n/2 the TRACE-FL aggregate norm is bounded by the largest honest norm."""
    U = honest_updates()
    byz = np.zeros(len(U), bool); byz[:10] = True
    honest_max = np.linalg.norm(U[~byz], axis=1).max()
    U[byz] = 1e4 * np.random.default_rng(1).normal(size=(10, U.shape[1]))
    g = tracefl(U, {})
    assert np.linalg.norm(g) <= honest_max + 1e-4


def test_collusion_factor_zeroes_duplicates():
    U = honest_updates()
    byz = np.zeros(len(U), bool); byz[:8] = True
    apply_attack("ipm", U, byz, np.random.default_rng(0))
    st = {}
    tracefl(U, st)
    assert st["rep"][byz].max() == 0.0 and st["rep"][~byz].mean() > 0


def test_foolsgold_weights_range():
    a = foolsgold_weights(honest_updates())
    assert a.min() >= 0 and a.max() <= 1


def test_bulyan_not_applicable():
    assert get_aggregator("bulyan")(honest_updates(n=40), {}, f=16) is None


def test_fixed_noise_attack_is_persistent():
    U1, U2 = honest_updates(seed=1), honest_updates(seed=2)
    byz = np.zeros(len(U1), bool); byz[:6] = True
    state, rng = {}, np.random.default_rng(0)
    apply_attack("aalief", U1, byz, rng, state)
    xi1 = state["xi"].copy()
    apply_attack("aalief", U2, byz, rng, state)
    assert np.array_equal(xi1, state["xi"])


def test_dirichlet_split_partitions_all_samples():
    y = np.repeat(np.arange(10), 50)
    parts = dirichlet_split(y, 20, 0.5, np.random.default_rng(0))
    allidx = np.sort(np.concatenate(parts))
    assert np.array_equal(allidx, np.arange(len(y)))


def test_mlp_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    m = MLP(5, 4, 3); w = m.init(rng).astype(np.float64)
    X = rng.normal(size=(6, 5)); y = rng.integers(0, 3, 6)

    def loss(v):
        Z = m.logits(v, X); Z = Z - Z.max(1, keepdims=True)
        return float(-(Z[np.arange(6), y] - np.log(np.exp(Z).sum(1))).mean())
    g = m.grad(w, X, y); i = 7; e = 1e-5
    wp, wm = w.copy(), w.copy(); wp[i] += e; wm[i] -= e
    assert abs((loss(wp) - loss(wm)) / (2 * e) - g[i]) < 1e-4


def test_local_update_reduces_loss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8)).astype(np.float32); y = (X[:, 0] > 0).astype(int)
    m = MLP(8, 16, 2); w = m.init(rng)
    acc0 = m.accuracy(w, X, y)
    w2 = w + local_update(m, w, X, y, rng, 0.2, 20, 16)
    assert m.accuracy(w2, X, y) >= max(acc0, 0.8)
