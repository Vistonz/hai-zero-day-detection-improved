# Seven improvements, tested

Each item below was run against the reproduced pipeline, not argued from the report.
Artifacts are in `results/`: `fusion_ablation.csv`, `null_event_recall.csv`,
`pr_auc_ci.csv`, `paired_tests.csv`, `attribution_process_check.csv`,
`calibration_miss.csv`, `attribution_ranks.csv`.

---

## ① The in-domain calibration already misses its own target

The EVT layer is calibrated on held-out source-normal windows to 10 alarms/day. On
the **in-domain 21.03 test files** the observed rate is:

| model | source calibration | 21.03 test | ratio to target |
|---|---|---|---|
| PCA | 9.7 /day | 30.1 /day | **3.0×** |
| Deep SVDD | 13.5 /day | 74.0 /day | **7.4×** |
| VAE | 10.6 /day | 18.3 /day | **1.8×** |

The calibration hits its target on the source split, then misses it by 2–7× on a
different period of the *same plant version*. This reframes the study: the
cross-version failure is not a clean transition from "works" to "fails". The
decision layer was already mis-calibrated in-domain, and the version shift
amplifies an existing gap. The report attributes the failure entirely to the shift.

---

## ② Max fusion is **not** the cause of the alarm collapse

Hypothesis: max fusion lets one drifting sensor alarm continuously. Re-running the
same calibration and evaluation with four fusion rules, on the saved per-sensor
scores (23.05, frozen arm, mean over seeds):

| model | fusion | PR-AUC | event recall | FA/day |
|---|---|---|---|---|
| PCA | max | 0.110 | 1.000 | 8297 |
| PCA | mean | 0.128 | 1.000 | 8297 |
| PCA | median | 0.121 | 1.000 | 8297 |
| PCA | top3mean | 0.106 | 1.000 | 8196 |
| Deep SVDD | max | 0.092 | 1.000 | 8297 |
| Deep SVDD | mean | 0.136 | 1.000 | 8297 |
| Deep SVDD | median | 0.039 | 0.513 | 1629 |
| Deep SVDD | top3mean | 0.094 | 1.000 | 8297 |
| VAE | max | 0.232 | 0.154 | 16.4 |
| VAE | mean | 0.247 | 0.154 | 14.9 |
| VAE | median | 0.108 | 0.032 | 8.0 |
| VAE | top3mean | 0.243 | 0.154 | 12.1 |

**The collapse survives every fusion rule.** PCA and Deep SVDD alarm continuously
under max, mean and top3-mean; only median changes their behaviour, and that is a
scale artefact, not a fix (Deep SVDD's PR-AUC falls to 0.039). Mean fusion does
improve cross-domain PR-AUC modestly — 0.110→0.128, 0.092→0.136, 0.232→0.247 — so it
is worth adopting, but it does not explain the failure.

The hypothesis was reasonable and it is wrong. The cause is the representation.

---

## ③ The event-recall metric, measured against its chance level

A circular shift rolls each model's alarm series within its stream, preserving the
run structure and destroying only temporal alignment with the attacks; 200 shifts per
configuration.

| model | domain | observed | null mean | p |
|---|---|---|---|---|
| PCA | 21.03 | 0.740 | 0.032 | 0.000 |
| Deep SVDD | 21.03 | 0.787 | 0.055 | 0.000 |
| VAE | 21.03 | 0.560 | 0.020 | 0.000 |
| PCA | 23.05 | 1.000 | 1.000 | **1.000** |
| Deep SVDD | 23.05 | 1.000 | 1.000 | **1.000** |
| VAE | 23.05 | 0.154 | 0.020 | 0.000 |

This turns the report's screening rule into a measurement. On 23.05, PCA and Deep
SVDD's perfect recall is **exactly what a randomly time-shifted alarm series
achieves** — every one of 200 shifts matched or beat it. VAE's 0.154, though low, is
7.7× its null and highly significant.

The report's phrase "these high-recall conditions are alarm-collapse failures" is
right; this makes it a statistical statement rather than a judgement call.

---

## ④ Confidence intervals and paired tests

**PR-AUC, 95% block-bootstrap interval** (600-sample blocks ≈ 100 min):

| model | 21.03 | 23.05 |
|---|---|---|
| PCA | 0.502 [0.404, 0.607] | 0.110 [0.037, 0.183] |
| Deep SVDD | 0.486 [0.383, 0.598] | 0.092 [0.028, 0.157] |
| VAE | 0.462 [0.372, 0.556] | 0.232 [0.068, 0.381] |

No in-domain interval overlaps its cross-domain counterpart: the degradation is
significant, not just a shift in the mean.

**Paired sign-flip tests on event recall**, paired by attack across seeds:

| domain | comparison | Δ recall | p |
|---|---|---|---|
| 21.03 | PCA vs Deep SVDD | −0.047 | 0.015 |
| 21.03 | PCA vs VAE | +0.180 | 0.000 |
| 21.03 | Deep SVDD vs VAE | +0.227 | 0.000 |
| 23.05 | PCA vs Deep SVDD | 0.000 | 1.000 |
| 23.05 | PCA vs VAE | +0.846 | 0.000 |

In-domain the ranking is significant. On 23.05, PCA and Deep SVDD are **identical**
(p = 1.000). The report says the two collapse; this shows they collapse to the same
point.

---

## ⑤ The hand-transcribed targets, checked against the CSV process labels

Each documented target's process (P1/P2/P3) was compared with the `attack_P1/P2/P3`
columns over that attack's interval:

**49 of 50 attacks agree (98%).**

The single exception is **A220**: the documentation lists `P2_LCV01D` while the CSV
labels P3 and P4. That is the same anomaly an independent parse of the technical PDF
produces — so it is an internal inconsistency in the documentation, not a
transcription error. Recording this one case is more useful than silently dropping it.

Note: 23.05 ships a single `label` column with no per-process labels, so this check
can only be run on 21.03.

---

## ⑥ Scores: a correction

`results/scores/` **does** contain per-window fused scores and per-sensor scores for
every model, seed and stream (90 files for the original run, 150 after the extra
seeds). My earlier claim that scores are not saved was wrong: they are produced, and
they are excluded from the repository by `.gitignore`. The pipeline is therefore
independently recomputable; what is missing is only the distribution.

Recommendation: ship a small archive of `results/scores/` (a few hundred MB
compressed) or document the exact command that regenerates it.

---

## ⑦ Smaller items

**Seeds: 3 → 5 changes almost nothing.** Running seeds 41 and 53 as well:

| model | metric | 3 seeds | 5 seeds |
|---|---|---|---|
| Deep SVDD | 21.03 recall | 0.787±0.012 | 0.784±0.009 |
| VAE | 21.03 recall | 0.560±0.000 | 0.560±0.000 |
| VAE | 23.05 PR-AUC | 0.232±0.000 | 0.232±0.000 |

The VAE's seed standard deviation is **below 0.001** — the five seeds converge to
essentially the same model. Extra seeds are cheap but buy very little here, and the
report's "three seeds" limitation is less serious than it reads.

**Attribution: median rank adds what Hit@3 hides.** Median rank of the best documented
target (41 sensors, chance 21):

| model | 21.03 | 23.05 |
|---|---|---|
| PCA | 8.0 | 17.5 |
| Deep SVDD | 6.0 | 15.0 |
| VAE | 7.0 | 19.0 |

PCA's Hit@3 on 23.05 is 0.033, which reads as useless; its median rank of 17.5 is
still better than chance. Hit@3 alone overstates the failure.

Broken down by target type on 23.05: attacks on actuators and sensors together
localise at rank 10, sensor+setpoint at 13, actuator+setpoint at 21.5, and
**actuator-only at 25.0 — worse than chance**. That is a mechanism-level finding the
Hit@3 rate cannot show.

**PCA's ±0.000 is exact, not noise.** PCA is deterministic: every metric has a seed
standard deviation below 1e-6. Reporting ±0.000 is correct but should be annotated.

**A latent dtype bug.** `results/metrics.csv` and `per_attack.csv` write `domain` and
`version` as `float64`, because "21.03" parses as a number. Any downstream filter
`df[df.domain == "21.03"]` silently returns an empty frame — it cost this analysis one
debugging cycle. Write those columns as strings.

**README wording.** "Reproduction of Group 88's HAI 21.03 → 23.05 study" — the authors
are Group 88. This is the group's own study, implementing its planning report.
