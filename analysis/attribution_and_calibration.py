"""Validate the hand-transcribed targets, and quantify the in-domain calibration miss.

Two questions:
  1. Do the documented attack targets agree with the CSV's per-process attack labels?
     (Only 21.03 carries process labels; 23.05 ships a single label column.)
  2. How far from its 10-alarms/day target does the EVT calibration land on the
     in-domain test set?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hai_zd.attacks import ATTACK_TARGETS
from hai_zd.data import FILES, load_stream
from hai_zd.metrics import FILE_ATTACKS, binary_events

RAW = ROOT / "data/raw"
PROC = {"P1": "attack_P1", "P2": "attack_P2", "P3": "attack_P3", "P4": "attack_P3"}

print("=" * 96)
print("1. DO THE DOCUMENTED TARGETS AGREE WITH THE CSV PROCESS LABELS? (21.03)")
print("=" * 96)
rows = []
for name in FILES["21.03"]["test"]:
    df = pd.read_csv(RAW / "hai-21.03" / f"{name}.csv.gz",
                     usecols=["attack", "attack_P1", "attack_P2", "attack_P3"])
    lab = (df["attack"].to_numpy() != 0)
    events = binary_events(lab.astype(np.uint8))
    ids = FILE_ATTACKS["21.03"][name]
    assert len(events) == len(ids), (len(events), len(ids))
    for aid, (a, b) in zip(ids, events):
        labelled = {p for p, col in PROC.items()
                    if (df[col].to_numpy()[a:b + 1] != 0).any()}
        documented = {t.split("_")[0] for t in ATTACK_TARGETS["21.03"][aid]}
        rows.append({"file": name, "attack_id": aid,
                     "csv_processes": "|".join(sorted(labelled)),
                     "target_processes": "|".join(sorted(documented)),
                     "agree": bool(labelled & documented)})
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results/attribution_process_check.csv", index=False)
print(f"  attacks checked: {len(t)}")
print(f"  documented target shares a process with the CSV labels: "
      f"{t.agree.sum()}/{len(t)} ({t.agree.mean():.1%})")
bad = t[~t.agree]
if len(bad):
    print("  disagreements:")
    for _, r in bad.iterrows():
        print(f"    {r.file} {r.attack_id}: targets={r.target_processes}  csv={r.csv_processes}")

print()
print("=" * 96)
print("2. IN-DOMAIN CALIBRATION: target 10 alarms/day, observed on the 21.03 test set")
print("=" * 96)
m = pd.read_csv(ROOT / "results/metrics.csv", dtype={"domain": str})
frozen = m[(m.domain == "21.03") & (m.arm == "frozen")]
target = 10.0
rows = []
for model, g in frozen.groupby("model"):
    fa = g.false_alarms_per_day.mean()
    src = g.calibration_alarms_per_day.mean()
    rows.append({"model": model, "source_calibration_fa": src,
                 "in_domain_test_fa": fa, "ratio_to_target": fa / target})
    print(f"  {model:5s}  source split {src:5.1f}/day   in-domain test {fa:6.1f}/day "
          f"({fa/target:.1f}x target)")
pd.DataFrame(rows).to_csv(ROOT / "results/calibration_miss.csv", index=False)

print()
print("=" * 96)
print("3. ATTRIBUTION: median target rank and breakdown by target-sensor type")
print("=" * 96)
a = pd.read_csv(ROOT / "results/per_attack.csv", dtype={"domain": str})
a = a[(a.arm == "frozen") & (a.best_target_rank.notna())]

def tag_kind(tags: str) -> str:
    kinds = set()
    for tag in str(tags).split("|"):
        if tag.endswith("NaN") or not tag or tag == "nan":
            continue
        if "_B" in tag or "AutoSD" in tag or "RTR" in tag or "VTR" in tag:
            kinds.add("setpoint")
        elif any(k in tag for k in ("FCV", "LCV", "PCV", "SCO", "LCP")):
            kinds.add("actuator")
        else:
            kinds.add("sensor")
    return "+".join(sorted(kinds)) if kinds else "unknown"

a = a.assign(kind=a.documented_targets.map(tag_kind))
print("  median rank of the best documented target (of 41 sensors, chance 21):")
for (model, domain), g in a.groupby(["model", "domain"]):
    print(f"    {model:5s} {domain}: median {g.best_target_rank.median():5.1f}  n={len(g)}")
print()
print("  by target-sensor type (pooled over models and seeds, 23.05):")
sub = a[a.domain == "23.05"]
for kind, g in sub.groupby("kind"):
    print(f"    {kind:22s} median rank {g.best_target_rank.median():5.1f}  n={len(g)}")
a[["model", "seed", "domain", "attack_id", "best_target_rank", "kind"]].to_csv(
    ROOT / "results/attribution_ranks.csv", index=False)
