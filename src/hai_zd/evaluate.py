"""Calibrate EVT thresholds on normal source data and evaluate both deployment arms."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

Array = NDArray[Any]

from .data import FILES, Stream, evaluation_ends, load_stream, load_tags
from .evt import DSPOT, SPOT
from .metrics import ScoreRecord, evaluate_records, per_attack_rows


@dataclass
class Calibration:
    q: float
    observed_alarms_per_day: float
    state: SPOT | DSPOT
    scored_samples: int


def _load_arrays(root: Path, model: str, seed: int, version: str, name: str) -> dict[str, Any]:
    path = root / "scores" / f"{model}_seed{seed}" / f"{version}_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing score artifact: {path}; run hai-train first")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _record(
    root: Path,
    processed_dir: Path,
    model: str,
    seed: int,
    version: str,
    name: str,
    window: int,
    stride: int,
    calibration: bool = False,
) -> tuple[Stream, dict[str, Any]]:
    stream = load_stream(processed_dir, version, name)
    arrays = _load_arrays(root, model, seed, version, name)
    expected = evaluation_ends(stream, window, stride, calibration=calibration)
    if not np.array_equal(arrays["ends"], expected):
        raise ValueError(f"Window endpoints mismatch in {model}_seed{seed}/{version}_{name}")
    n = len(expected)
    if arrays["scores"].shape != (n,) or arrays["sensor_scores"].shape != (n, stream.x.shape[1]):
        raise ValueError(f"Score shape mismatch in {model}_seed{seed}/{version}_{name}")
    if not np.isfinite(arrays["scores"]).all() or not np.isfinite(arrays["sensor_scores"]).all():
        raise ValueError(f"Non-finite scores in {model}_seed{seed}/{version}_{name}")
    return stream, arrays


def _new_threshold(method: str, initial: Array, q: float, config: dict[str, Any]) -> SPOT | DSPOT:
    common = {
        "q": q,
        "level": float(config["threshold"]["initial_quantile"]),
        "min_peaks": int(config["threshold"]["min_exceedances"]),
        "update_every": int(config["threshold"]["update_every"]),
    }
    if method == "frozen":
        return SPOT(initial, **common)
    return DSPOT(initial, depth=int(config["threshold"]["dspot_depth"]), **common)


def calibrate(
    scores: Array,
    method: str,
    config: dict[str, Any],
) -> Calibration:
    stride = int(config["window"]["eval_stride"])
    target = float(config["threshold"]["alarms_per_day"])
    init_count = min(int(config["threshold"]["initialization_samples"]), len(scores) // 3)
    if init_count < 100:
        raise ValueError("Not enough normal calibration scores")
    initial, scored = scores[:init_count], scores[init_count:]
    nominal_q = target * stride / 86400.0
    grid_size = max(int(config["threshold"]["calibration_grid_size"]), 3)
    grid = np.geomspace(nominal_q * 1e-6, min(0.01, nominal_q * 1e3), num=grid_size)
    candidates = sorted({nominal_q, *grid.tolist()})
    best: tuple[float, float, float, SPOT | DSPOT] | None = None
    for q in candidates:
        state = _new_threshold(method, initial, q, config)
        result = state.run(scored)
        rate = float(result.alarms.sum()) * 86400.0 / (len(scored) * stride)
        candidate = (abs(rate - target), q, rate, state)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("EVT calibration did not evaluate any candidate")
    _, q, rate, state = best
    return Calibration(q=q, observed_alarms_per_day=rate, state=state, scored_samples=len(scored))


def _save_predictions(
    root: Path,
    model: str,
    seed: int,
    version: str,
    name: str,
    arm: str,
    ends: Array,
    predictions: Array,
    thresholds: Array,
) -> None:
    output = root / "predictions" / f"{model}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{version}_{name}_{arm}.npz",
        ends=ends,
        predictions=predictions.astype(np.uint8),
        thresholds=thresholds,
    )


def evaluate(config: dict[str, Any], root: Path, models: list[str], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(config["data"]["processed_dir"])
    window = int(config["window"]["length"])
    stride = int(config["window"]["eval_stride"])
    top_k = int(config["evaluation"]["attribution_top_k"])
    tags = load_tags(processed_dir)
    metric_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            calibration_parts = []
            for name in FILES["21.03"]["train"]:
                _, arrays = _record(root, processed_dir, model, seed, "21.03", name, window, stride, True)
                calibration_parts.append(arrays["scores"])
            normal_scores = np.concatenate(calibration_parts)
            calibrations = {arm: calibrate(normal_scores, arm, config) for arm in ("frozen", "adaptive")}
            for arm, calibration in calibrations.items():
                calibration_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "arm": arm,
                        "q": calibration.q,
                        "observed_alarms_per_day": calibration.observed_alarms_per_day,
                        "target_alarms_per_day": float(config["threshold"]["alarms_per_day"]),
                        "scored_samples": calibration.scored_samples,
                        "source_threshold": float(calibration.state.threshold),
                    }
                )
            for version in ("21.03", "23.05"):
                for arm, calibration in calibrations.items():
                    records: list[ScoreRecord] = []
                    for name in FILES[version]["test"]:
                        stream, arrays = _record(root, processed_dir, model, seed, version, name, window, stride)
                        if arm == "frozen":
                            threshold = float(calibration.state.threshold)
                            predictions = arrays["scores"] > threshold
                            thresholds = np.full(len(predictions), threshold, dtype=np.float64)
                        else:
                            result = copy.deepcopy(calibration.state).run(arrays["scores"])
                            predictions, thresholds = result.alarms, result.thresholds
                        _save_predictions(root, model, seed, version, name, arm, arrays["ends"], predictions, thresholds)
                        records.append(ScoreRecord(stream, arrays["ends"], arrays["scores"], arrays["sensor_scores"], predictions))
                    attacks = per_attack_rows(records, tags, top_k, model, seed, arm)
                    metrics = evaluate_records(records, stride, attacks)
                    metric_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "domain": version,
                            "arm": arm,
                            "q": calibration.q,
                            "calibration_alarms_per_day": calibration.observed_alarms_per_day,
                            **metrics,
                        }
                    )
                    attack_rows.extend(attacks)
                    print(
                        f"{model} seed={seed} {version} {arm}: "
                        f"detection={metrics['attack_detection_rate']:.3f} "
                        f"PR-AUC={metrics['pr_auc']:.3f} FA/day={metrics['false_alarms_per_day']:.2f}"
                    )
    metrics_frame = pd.DataFrame(metric_rows).sort_values(["model", "seed", "domain", "arm"])
    attacks_frame = pd.DataFrame(attack_rows).sort_values(["model", "seed", "arm", "domain", "file", "attack_id"])
    # Keep version labels as text: "21.03" otherwise parses back as a float and every
    # downstream `df[df.domain == "21.03"]` filter silently returns nothing.
    for frame in (metrics_frame, attacks_frame):
        for column in ("domain", "version"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    root.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(root / "metrics.csv", index=False)
    attacks_frame.to_csv(root / "per_attack.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(root / "calibration.csv", index=False)
    (root / "evaluation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return metrics_frame, attacks_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--model", choices=["pca", "svdd", "vae", "all"], default="all")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = ["pca", "svdd", "vae"] if args.model == "all" else [args.model]
    seeds = args.seed or [int(seed) for seed in config["training"]["seeds"]]
    evaluate(config, args.root, models, seeds)


if __name__ == "__main__":
    main()
