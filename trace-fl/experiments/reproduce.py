"""Reproduce the tables of the TRACE-FL paper.

  python experiments/reproduce.py --list                    # tables and job counts
  python experiments/reproduce.py --table table3            # one table
  python experiments/reproduce.py --table all --workers 4   # everything, 4 processes

Every finished run is cached in results/runs.json, so interrupted jobs resume and
tables that share runs are not recomputed. Table numbers follow the manuscript.
Approximate single-core CPU time: Fashion-MNIST ~8 s/run, Digits ~1 s/run,
CIFAR-10 features ~40-60 s/run (features are built once, ~3-5 min).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tracefl import run_fl  # noqa: E402

CACHE = os.path.join(ROOT, "results", "runs.json")
STD = ["none", "lflip", "sign", "gauss", "alie", "ipm", "scale"]
STD_ATT = STD[1:]
NAMES = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median", "trmean": "TrMean", "cclip": "CClip",
         "fltrust": "FLTrust*", "bulyan": "Bulyan", "foolsgold": "FoolsGold", "rfa": "RFA",
         "tracefl": "TRACE-FL", "tracefl_plus": "TRACE-FL+"}
ATT = {"none": "No attack", "lflip": "LF", "sign": "SF", "gauss": "GS", "alie": "ALIE", "ipm": "IPM",
       "scale": "SLF", "aalie": "A-ALIE", "aipm": "A-IPM", "aalief": "A-ALIE-F", "aipmf": "A-IPM-F"}
ABL = {"Full TRACE-FL": {}, "w/o clipping": {"clip": False}, "w/o reputation": {"rep": False},
       "w/o distance": {"dist": False}, "w/o collusion": {"coll": False}, "w/o persistence": {"pers": False}}


def J(row, col, ds, agg, att, fb=0.2, seeds=(0,), alpha=0.5, kw=None, avg=True):
    return dict(row=row, col=col, ds=ds, agg=agg, att=att, fb=fb, seeds=list(seeds), alpha=alpha,
                kw=kw or {}, avg=avg)


def standard(ds, methods, seeds):
    return [J(NAMES[m], ATT[a], ds, m, a, seeds=seeds, avg=a != "none") for m in methods for a in STD]


def build_tables():
    base7 = ["fedavg", "krum", "median", "trmean", "cclip", "fltrust", "tracefl"]
    base6 = ["fedavg", "median", "trmean", "cclip", "fltrust", "tracefl"]
    cf_att = ["none", "sign", "alie", "ipm", "aalie", "aipm"]
    T = {}
    T["table3"] = ("Fashion-MNIST, standard attacks, 20% Byzantine", standard("fmnist", base7, (0, 1)))
    T["table4"] = ("UCI Digits, standard attacks, 20% Byzantine", standard("digits", base7, (0, 1, 2)))
    T["table5a"] = ("CIFAR-10 (seed 0)",
                    [J(NAMES[m], ATT[a], "cifar", m, a, avg=a != "none") for m in base6 for a in cf_att]
                    + [J(NAMES[m], ATT[a] + " 40%", "cifar", m, a, fb=0.4, avg=False)
                       for m in base6 for a in ("aalie", "aipm")])
    T["table5b"] = ("CIFAR-10, 3 seeds",
                    [J(NAMES[m], f"{ATT[a]} {int(fb * 100)}%", "cifar", m, a, fb=fb, seeds=(0, 1, 2), avg=False)
                     for m in ("fedavg", "cclip", "fltrust", "tracefl")
                     for a, fb in (("sign", .2), ("ipm", .2), ("aipm", .2), ("aipm", .4))])
    T["table6"] = ("Fashion-MNIST, adaptive attacks",
                   [J(NAMES[m], f"{ATT[a]} {int(fb * 100)}%", "fmnist", m, a, fb=fb, seeds=(0, 1), avg=False)
                    for m in base6 for fb in (.2, .4) for a in ("aalie", "aipm")]
                   + [J("TRACE-FL w/o persistence (v1)", f"{ATT[a]} {int(fb * 100)}%", "fmnist", "tracefl", a,
                        fb=fb, seeds=(0, 1), kw={"pers": False}, avg=False)
                      for fb in (.2, .4) for a in ("aalie", "aipm")])
    sw = ["median", "trmean", "cclip", "fltrust", "tracefl"]
    T["table7"] = ("Fashion-MNIST sweeps (seed 0): alpha under ALIE 20%, attacker fraction under IPM",
                   [J(NAMES[m], f"alpha={al}", "fmnist", m, "alie", alpha=al, avg=False) for m in sw for al in (.1, .5, 1.0)]
                   + [J(NAMES[m], f"IPM {int(fb * 100)}%", "fmnist", m, "ipm", fb=fb, avg=False)
                      for m in sw for fb in (.1, .2, .3, .4)])
    T["table8"] = ("Ablation, Fashion-MNIST, 40% Byzantine (seed 0); cells: acc / Byzantine weight share",
                   [J(v, ATT[a], "fmnist", "tracefl", a, fb=0.4, kw=kw, avg=False)
                    for v, kw in ABL.items() for a in ("sign", "alie", "ipm", "aalie", "aipm")])
    ex = ["foolsgold", "rfa", "bulyan", "tracefl"]
    T["table9"] = ("Extra baselines: Avg / Worst over attacks at 20%, and adaptive IPM at 40%",
                   standard("fmnist", ex, (0, 1)) + standard("digits", ex, (0, 1, 2))
                   + [J(NAMES[m], "CIFAR " + ATT[a], "cifar", m, a) for m in ex for a in cf_att[1:]]
                   + [J(NAMES[m], "A-IPM 40% FMNIST", "fmnist", m, "aipm", fb=.4, seeds=(0, 1), avg=False) for m in ex]
                   + [J(NAMES[m], "A-IPM 40% CIFAR", "cifar", m, "aipm", fb=.4, avg=False) for m in ex])
    tp = ["tracefl", "tracefl_plus"]
    T["table10"] = ("TRACE-FL vs TRACE-FL+",
                    standard("fmnist", tp, (0, 1)) + standard("digits", tp, (0, 1, 2))
                    + [J(NAMES[m], "CIFAR " + ATT[a], "cifar", m, a) for m in tp for a in cf_att[1:]]
                    + [J(NAMES[m], f"FMNIST {ATT[a]} 40%", "fmnist", m, a, fb=.4, seeds=(0, 1), avg=False)
                       for m in tp for a in ("aalie", "aipm")]
                    + [J(NAMES[m], f"CIFAR {ATT[a]} {int(fb * 100)}% (3 seeds)", "cifar", m, a, fb=fb,
                         seeds=(0, 1, 2), avg=False)
                       for m in tp for a, fb in (("sign", .2), ("ipm", .2), ("aipm", .2), ("aipm", .4))])
    pf = ["fedavg", "median", "trmean", "cclip", "fltrust", "foolsgold", "rfa", "tracefl", "tracefl_plus"]
    T["table11"] = ("Persistence-faking attacks, Fashion-MNIST",
                    [J(NAMES[m], f"{ATT[a]} {int(fb * 100)}%", "fmnist", m, a, fb=fb, seeds=(0, 1), avg=False)
                     for m in pf for fb in (.2, .4) for a in ("aalief", "aipmf")])
    return T


def key(j, seed):
    return json.dumps([j["ds"], j["agg"], j["att"], j["fb"], j["alpha"], seed, sorted(j["kw"].items())])


def _work(args):
    j, seed = args
    return key(j, seed), run_fl(j["ds"], j["agg"], j["att"], alpha=j["alpha"], frac_byz=j["fb"],
                                rounds=30, seed=seed, agg_kwargs=j["kw"])


def summarize(title, jobs, cache):
    rows, cols = [], []
    for j in jobs:
        rows += [j["row"]] if j["row"] not in rows else []
        cols += [j["col"]] if j["col"] not in cols else []
    grid = {}
    for j in jobs:
        rs = [cache.get(key(j, s)) for s in j["seeds"]]
        if any(r is None for r in rs):
            grid[(j["row"], j["col"])] = ("n/a", None, j["avg"]) if all(r is None for r in rs) else ("…", None, j["avg"])
            continue
        a = np.array([r["acc"] for r in rs]) * 100
        cell = f"{a.mean():.1f}" + (f"±{a.std():.1f}" if len(a) > 1 else "")
        if "rep_byz" in rs[0] and j["kw"] is not None and j["row"] in ABL:
            nb = int(j["fb"] * 40)
            sh = np.mean([100 * nb * r["rep_byz"] / (nb * r["rep_byz"] + (40 - nb) * r["rep_ben"] + 1e-30) for r in rs])
            cell += f" / {sh:.1f}%"
        grid[(j["row"], j["col"])] = (cell, a.mean(), j["avg"])
    show_avg = any(v[2] for v in grid.values())
    head = ["Method"] + cols + (["Avg", "Worst"] if show_avg else [])
    print(f"\n### {title}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        line = [r] + [grid.get((r, c), ("", None, False))[0] for c in cols]
        if show_avg:
            vals = [grid[(r, c)][1] for c in cols if (r, c) in grid and grid[(r, c)][2]]
            ok = vals and all(v is not None for v in vals)
            line += [f"{np.mean(vals):.1f}", f"{min(vals):.1f}"] if ok else ["", ""]
        print("| " + " | ".join(line) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="table3", help="table3 ... table11, or 'all'")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    T = build_tables()
    if a.list:
        for k, (title, jobs) in T.items():
            print(f"{k:8s} {sum(len(j['seeds']) for j in jobs):4d} runs  {title}")
        return
    names = list(T) if a.table == "all" else [a.table]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo, seen = [], set()
    for n in names:
        for j in T[n][1]:
            for s in j["seeds"]:
                k = key(j, s)
                if k not in cache and k not in seen:
                    seen.add(k)
                    todo.append((j, s))
    print(f"{len(todo)} runs to compute ({len(cache)} cached)")
    if todo:
        pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
        it = as_completed([pool.submit(_work, x) for x in todo]) if pool else (_work(x) for x in todo)
        for i, res in enumerate(it, 1):
            k, r = res.result() if pool else res
            cache[k] = r
            json.dump(cache, open(CACHE, "w"))
            print(f"[{i}/{len(todo)}] {k} -> {'n/a' if r is None else round(100 * r['acc'], 2)}", flush=True)
    for n in names:
        summarize(f"{n}: {T[n][0]}", T[n][1], cache)


if __name__ == "__main__":
    main()
