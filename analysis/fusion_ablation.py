"""Does max fusion cause the cross-domain alarm collapse?

The pipeline fuses 41 per-sensor scores by maximum, so a single drifting sensor can
alarm continuously. This re-runs the same calibration and evaluation with four fusion
rules, reusing the saved per-sensor scores.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hai_zd.data import FILES, evaluation_ends, load_stream, load_tags
from hai_zd.evaluate import _load_arrays, calibrate
from hai_zd.metrics import ScoreRecord, evaluate_records, per_attack_rows

cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
proc = ROOT / cfg["data"]["processed_dir"]
res = ROOT / "results"
WIN, STRIDE = cfg["window"]["length"], cfg["window"]["eval_stride"]
TOP_K = cfg["evaluation"]["attribution_top_k"]
TAGS = load_tags(proc)
MODELS = ["pca", "svdd", "vae"]
SEEDS = cfg["training"]["seeds"]

def fuse(S: np.ndarray, how: str) -> np.ndarray:
    if how == "max":
        return S.max(axis=1)
    if how == "mean":
        return S.mean(axis=1)
    if how == "median":
        return np.median(S, axis=1)
    if how == "top3mean":
        return np.sort(S, axis=1)[:, -3:].mean(axis=1)
    raise ValueError(how)

FUSIONS = ["max", "mean", "median", "top3mean"]

def load(model, seed, version, name, calibration=False):
    stream = load_stream(proc, version, name)
    arrays = _load_arrays(res, model, seed, version, name)
    ends = evaluation_ends(stream, WIN, STRIDE, calibration=calibration)
    if not np.array_equal(arrays["ends"], ends):
        raise ValueError("endpoint mismatch")
    return stream, arrays

rows = []
for model in MODELS:
    for seed in SEEDS:
        cache = {}
        for version in ("21.03", "23.05"):
            names = FILES[version]["train"] if version == "21.03" else []
        # calibration streams
        cal_stems = [load(model, seed, "21.03", n, calibration=True) for n in FILES["21.03"]["train"]]
        test_stems = {v: [load(model, seed, v, n) for n in FILES[v]["test"]]
                      for v in ("21.03", "23.05")}
        for how in FUSIONS:
            normal = np.concatenate([fuse(a["sensor_scores"], how) for _, a in cal_stems])
            cals = {arm: calibrate(normal, arm, cfg) for arm in ("frozen", "adaptive")}
            for arm, cal in cals.items():
                for version, stems in test_stems.items():
                    records = []
                    for stream, arrays in stems:
                        s = fuse(arrays["sensor_scores"], how)
                        if arm == "frozen":
                            preds = s > float(cal.state.threshold)
                        else:
                            preds = copy.deepcopy(cal.state).run(s).alarms
                        records.append(ScoreRecord(stream, arrays["ends"], s,
                                                   arrays["sensor_scores"], preds))
                    attacks = per_attack_rows(records, TAGS, TOP_K, model, seed, arm)
                    m = evaluate_records(records, STRIDE, attacks)
                    rows.append({"model": model, "seed": seed, "fusion": how,
                                 "domain": version, "arm": arm,
                                 "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                                 "event_recall": m["attack_detection_rate"],
                                 "fa_per_day": m["false_alarms_per_day"],
                                 "eta_f1": m["eta_f1"],
                                 "normal_alarm_fraction": m.get("normal_window_alarm_fraction",
                                                                float("nan"))})
        print(f"done {model} seed={seed}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(res / "fusion_ablation.csv", index=False)
print("\n" + "=" * 92)
print("FUSION ABLATION: cross-domain (23.05) frozen arm, mean over seeds")
print("=" * 92)
g = df[(df.domain == "23.05") & (df.arm == "frozen")].groupby(["model", "fusion"])[
    ["pr_auc", "event_recall", "fa_per_day"]].mean()
for model in MODELS:
    print(f"\n{model}")
    print(f"  {'fusion':10s} {'PR-AUC':>7s} {'recall':>7s} {'FA/day':>10s}")
    for how in FUSIONS:
        r = g.loc[(model, how)]
        print(f"  {how:10s} {r.pr_auc:7.3f} {r.event_recall:7.3f} {r.fa_per_day:10.1f}")
print("\n(in-domain 21.03, frozen arm)")
g2 = df[(df.domain == "21.03") & (df.arm == "frozen")].groupby(["model", "fusion"])[
    ["pr_auc", "event_recall", "fa_per_day"]].mean()
for model in MODELS:
    print(f"\n{model}")
    for how in FUSIONS:
        r = g2.loc[(model, how)]
        print(f"  {how:10s} PR-AUC {r.pr_auc:.3f}  recall {r.event_recall:.3f}  FA/day {r.fa_per_day:6.1f}")
