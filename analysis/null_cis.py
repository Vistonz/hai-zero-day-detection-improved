"""Chance level, confidence intervals, and paired tests for the version-shift study.

Three additions to the original analysis:
  * a circular-shift null for event recall, so "recall 1.000" can be read against
    what a randomly time-shifted version of the same alarm series would achieve;
  * block-bootstrap intervals for PR-AUC and segment-bootstrap intervals for event
    recall;
  * paired sign-flip permutation tests between models, pairing attacks across seeds.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hai_zd.data import FILES, evaluation_ends, load_stream
from hai_zd.metrics import binary_events

cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
proc = ROOT / cfg["data"]["processed_dir"]
res = ROOT / "results"
WIN, STRIDE = cfg["window"]["length"], cfg["window"]["eval_stride"]
MODELS = ["pca", "svdd", "vae"]
SEEDS = cfg["training"]["seeds"]
DOMAINS = ["21.03", "23.05"]

# ---------------------------------------------------------------- load predictions
def load_preds(model, seed, domain, arm):
    out = []
    for name in FILES[domain]["test"]:
        stream = load_stream(proc, domain, name)
        z = np.load(res / "predictions" / f"{model}_seed{seed}" /
                    f"{domain}_{name}_{arm}.npz")
        out.append({"stream": stream, "pred": z["predictions"].astype(bool),
                    "ends": z["ends"]})
    return out

def event_recall(parts, rng=None):
    """Event recall, using documented intervals mapped through window endpoints.

    A circular shift rolls the alarm series within each stream, preserving its run
    structure and only destroying temporal alignment with the attacks.
    """
    hit = tot = 0
    for p in parts:
        pred, ends = p["pred"], p["ends"]
        if rng is not None:
            pred = np.roll(pred, int(rng.integers(1, len(pred))))
        for a, b in binary_events(p["stream"].y):
            positions = np.flatnonzero((ends >= a) & (ends <= b))
            tot += 1
            if len(positions) and pred[positions].any():
                hit += 1
    return hit / tot if tot else float("nan")

# ---------------------------------------------------------------- 1. circular-shift null
print("=" * 96)
print("1. CIRCULAR-SHIFT NULL for event recall (200 shifts per configuration)")
print("=" * 96)
null_rows = []
for model in MODELS:
    for domain in DOMAINS:
        for arm in ("frozen", "adaptive"):
            obs, nulls = [], []
            for seed in SEEDS:
                parts = load_preds(model, seed, domain, arm)
                rng = np.random.default_rng(seed)
                obs.append(event_recall(parts))
                nulls.append([event_recall(parts, rng=rng) for _ in range(200)])
            o = float(np.mean(obs)); n = np.array(nulls).ravel()
            null_rows.append({"model": model, "domain": domain, "arm": arm,
                              "observed": o, "null_mean": float(n.mean()),
                              "null_p95": float(np.percentile(n, 95)),
                              "p_value": float((n >= o).mean())})
            print(f"  {model:5s} {domain} {arm:9s} observed={o:.3f} "
                  f"null={n.mean():.3f} p={(n >= o).mean():.3f}")
pd.DataFrame(null_rows).to_csv(res / "null_event_recall.csv", index=False)

# ---------------------------------------------------------------- 2. PR-AUC intervals
print()
print("=" * 96)
print("2. PR-AUC WITH 95% BLOCK-BOOTSTRAP INTERVAL (block 600 samples = 100 min)")
print("=" * 96)
def concat_scores(model, seed, domain):
    S, Y = [], []
    for name in FILES[domain]["test"]:
        z = np.load(res / "scores" / f"{model}_seed{seed}" / f"{domain}_{name}.npz")
        S.append(z["scores"]); Y.append(z["y"].astype(int))
    return np.concatenate(S), np.concatenate(Y)

pr_rows = []
for model in MODELS:
    for domain in DOMAINS:
        obs, lo, hi = [], [], []
        for seed in SEEDS:
            s, y = concat_scores(model, seed, domain)
            obs.append(average_precision_score(y, s))
            rng = np.random.default_rng(seed)
            n = len(s); block = 600; nb = int(np.ceil(n / block))
            bounds = [(i * block, min((i + 1) * block, n)) for i in range(nb)]
            vals = np.empty(400)
            for b in range(400):
                pick = rng.integers(0, nb, nb)
                idx = np.concatenate([np.arange(*bounds[i]) for i in pick])
                vals[b] = average_precision_score(y[idx], s[idx]) if len(np.unique(y[idx])) > 1 else np.nan
            v = vals[np.isfinite(vals)]
            lo.append(np.percentile(v, 2.5)); hi.append(np.percentile(v, 97.5))
        pr_rows.append({"model": model, "domain": domain, "pr_auc": float(np.mean(obs)),
                        "ci_lo": float(np.mean(lo)), "ci_hi": float(np.mean(hi))})
        r = pr_rows[-1]
        print(f"  {model:5s} {domain}: {r['pr_auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
pd.DataFrame(pr_rows).to_csv(res / "pr_auc_ci.csv", index=False)

# ---------------------------------------------------------------- 3. paired tests
print()
print("=" * 96)
print("3. PAIRED SIGN-FLIP TESTS on event recall, paired by attack across seeds")
print("=" * 96)
# domain/version are written as floats ("21.03" parses as a number); force strings,
# otherwise every string filter below silently selects nothing.
att = pd.read_csv(res / "per_attack.csv",
                  dtype={"domain": str, "version": str})
frozen = att[att.arm == "frozen"]
def per_attack_vector(model, domain):
    g = frozen[(frozen.model == model) & (frozen.domain == domain)].copy()
    g["det"] = g["detected"].astype(str).str.lower().eq("true").astype(float)
    return g.groupby(["seed", "file", "attack_id"]).det.mean().sort_index()

test_rows = []
for domain in DOMAINS:
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            va, vb = per_attack_vector(a, domain), per_attack_vector(b, domain)
            idx = va.index.intersection(vb.index)
            da, db = va.loc[idx].to_numpy(), vb.loc[idx].to_numpy()
            diff = float((da - db).mean())
            rng = np.random.default_rng(0)
            null = np.array([np.mean((da - db) * rng.choice([-1, 1], len(da))) for _ in range(4000)])
            p = float((np.abs(null) >= abs(diff)).mean())
            test_rows.append({"domain": domain, "a": a, "b": b,
                              "delta_event_recall": diff, "p": p})
            print(f"  {domain}  {a:5s} vs {b:5s}  delta={diff:+.3f}  p={p:.3f}")
pd.DataFrame(test_rows).to_csv(res / "paired_tests.csv", index=False)
print("\nwrote null_event_recall.csv, pr_auc_ci.csv, paired_tests.csv")
