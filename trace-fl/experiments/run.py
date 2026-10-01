"""Run a single federated-learning configuration.

Examples
  python experiments/run.py --dataset fmnist --agg tracefl_plus --attack aipm --byz 0.4 --seeds 0 1
  python experiments/run.py --dataset digits --agg cclip --attack sign
  python experiments/run.py --dataset fmnist --agg tracefl --attack alie --no-pers   # ablation
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracefl import AGGREGATORS, ATTACKS, run_fl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "digits", "cifar"])
    ap.add_argument("--agg", default="tracefl_plus", help=f"one of {sorted(AGGREGATORS)}")
    ap.add_argument("--attack", default="none", choices=ATTACKS)
    ap.add_argument("--byz", type=float, default=0.2, help="fraction of Byzantine clients")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration")
    ap.add_argument("--clients", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gamma", type=float, default=0.8, help="TRACE-FL reputation smoothing")
    for comp in ("clip", "rep", "dist", "coll", "pers"):
        ap.add_argument(f"--no-{comp}", action="store_true", help=f"TRACE-FL ablation: disable {comp}")
    ap.add_argument("--json", action="store_true", help="print raw JSON results")
    a = ap.parse_args()

    kw = {}
    if a.agg.startswith("trace"):
        kw["gamma"] = a.gamma
        for comp in ("clip", "rep", "dist", "coll", "pers"):
            if getattr(a, f"no_{comp}"):
                kw[comp] = False

    res = []
    for s in a.seeds:
        r = run_fl(a.dataset, a.agg, a.attack, alpha=a.alpha, n_clients=a.clients, frac_byz=a.byz,
                   rounds=a.rounds, seed=s, agg_kwargs=kw)
        if r is None:
            print(f"{a.agg} is not applicable with {a.byz:.0%} Byzantine clients (needs n >= 4f + 3)")
            return
        res.append(r)
        extra = ""
        if "rep_byz" in r:
            nb = int(a.byz * a.clients)
            tot = nb * r["rep_byz"] + (a.clients - nb) * r["rep_ben"]
            extra = f"   Byzantine weight share {100 * nb * r['rep_byz'] / tot if tot else 0:.1f}%"
        print(f"seed {s}: accuracy {100 * r['acc']:.2f}%{extra}", flush=True)
    accs = np.array([r["acc"] for r in res]) * 100
    print(f"{a.dataset} | {a.agg} | {a.attack} | {a.byz:.0%} Byzantine | alpha {a.alpha}: "
          f"{accs.mean():.2f} +- {accs.std():.2f} (n={len(accs)})")
    if a.json:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
