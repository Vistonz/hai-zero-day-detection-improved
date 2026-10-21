# Unsupervised Zero-Day Attack Detection in an ICS

Implementation of Group 88's HAI 21.03 → 23.05 planning report: matched-capacity Deep SVDD and convolutional VAE, a PCA reconstruction baseline, per-sensor attribution, and SPOT/DSPOT decision layers under plant-version shift.

## Protocol

- Train representations only on normal HAI 21.03 data; never use attack labels for training, model selection, or threshold calibration.
- Start with 59 tags shared by HAI 21.03 and 23.05, then retain 41 tags with non-zero variance on the source training partitions. This source-only rule prevents incompatible constant encodings from dominating max fusion. Fit normalization on the chronological training partitions only.
- Evaluate unseen attacks in-domain (21.03, 50 events) and cross-version (23.05, 52 events).
- Score causal 60-second windows every 10 seconds, aligned to the window endpoint. Fuse per-sensor scores by maximum.
- Calibrate each model/seed on source normal data at 10 false-alarm points/day. Compare frozen SPOT with test-stream DSPOT.
- Report PR-AUC, ROC-AUC, unadjusted point metrics, eTaPR, event detection/delay, false alarms/day, and per-attack target attribution. No point adjustment is used.
- Default experiment: three seeds (11, 23, 37).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

For a GPU environment, install the appropriate PyTorch build instead. The supplied experiments run on CPU.

## Reproduce

```bash
# Verify/download the exact 341 MB raw dataset and technical documentation.
hai-download

# Align tags, parse both label formats, fit source normalization, and cache streams.
hai-prepare

# Optional fast end-to-end check; writes only to results/smoke.
hai-train --model all --seed 11 --smoke

# Full grid: PCA, Deep SVDD, and VAE × three seeds, followed by all stream scores.
hai-train --model all

# Source-only EVT calibration and both deployment arms on both test domains.
hai-evaluate

# Aggregate statistics, figures, final report, and poster (Markdown/PDF/PNG).
hai-report

# Behavioral regression tests.
pytest
```

Individual models or seeds can be rerun with `--model {pca,svdd,vae}` and `--seed N` on the training/evaluation commands.

## Outputs

- `data/processed/metadata.json`: row counts, attack counts, label alignment, source SHA-256.
- `results/checkpoints/`: model state and training configuration/history.
- `results/scores/`: per-sensor and fused scores for every stream.
- `results/metrics.csv`, `per_attack.csv`, `calibration.csv`: unaggregated experimental evidence.
- `results/metrics_summary.csv`, `degradation_recovery.csv`, `recovery_summary.csv`: three-seed analysis.
- `figures/`: degradation, PR-AUC, false-alarm, delay, attribution, and per-attack plots.
- `reports/final_report.{md,pdf}` and `reports/poster.{md,pdf,png}`.

## Data and references

HAI is distributed under CC BY-SA 4.0 at <https://github.com/icsdataset/hai>. The downloader pins the current official files by SHA-256. HAI 23.05 `label-test2.csv` has minute-resolution timestamps; the loader accepts them only after verifying every value against the corresponding floored telemetry timestamp.

The report cites Deep SVDD (ICML 2018), SPOT/DSPOT (KDD 2017), TaPR (CIKM 2019), eTaPR (SAC 2022), and the time-series evaluation protocol of Kim et al. (AAAI 2022).

## Additions in this version

This version adds a statistical-check layer on top of the original study. It reuses
the saved per-window and per-sensor scores; no model was retrained for it.

- `analysis/fusion_ablation.py` — reruns calibration and evaluation with max, mean,
  median and top-3-mean fusion, on the saved per-sensor scores.
- `analysis/null_cis.py` — circular-shift null for event recall, block-bootstrap
  intervals for PR-AUC, and paired sign-flip tests between models.
- `analysis/attribution_and_calibration.py` — validates the documented targets against
  the CSV process labels, quantifies the in-domain calibration miss, and reports the
  median target rank by target type.
- `analysis/IMPROVEMENTS.md` — all findings with tables and the commands that produced
  them.

What the additions show:

- **The cross-domain recall of 1.000 is alarm collapse, measured.** A circular shift of
  the alarm series reaches the same recall in 200 of 200 shifts (p = 1.000) for PCA and
  DeepSVDD on 23.05; on 21.03 the same models beat their null at p < 0.001.
- **The EVT calibration already misses its own 10 alarms/day target in-domain**, by
  1.8x (VAE) to 7.4x (DeepSVDD). The version shift amplifies an existing gap.
- **Max fusion is not the cause of the collapse.** PCA and DeepSVDD alarm continuously
  under max, mean and top-3-mean; only median changes behaviour, and that is a scale
  artefact.
- **The documented targets are reliable.** They agree with the CSV per-process labels
  in 49 of 50 attacks; the single exception, A220, is an internal inconsistency in the
  documentation.
- **Hit@3 understates attribution.** PCA's Hit@3 falls to 0.033 on 23.05, but its
  median target rank of 17.5 (chance 21) is still better than random.

`reports/final_report.md` and `.pdf` carry a new **Statistical Checks (added)**
section generated by `hai-report`. `poster/` holds an A2 poster built from the same
results.
