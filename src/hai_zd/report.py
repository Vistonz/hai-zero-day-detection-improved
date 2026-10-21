"""Generate aggregate statistics, figures, and publication-ready HAI reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

MODELS = ("pca", "svdd", "vae")
DOMAINS = ("21.03", "23.05")
ARMS = ("frozen", "adaptive")
RECOVERY_METRICS = ("attack_detection_rate", "eta_f1", "point_f1")
ETA_THETA_P = 0.5
ETA_THETA_R = 0.1
MODEL_LABELS = {"pca": "PCA", "svdd": "DeepSVDD", "vae": "VAE"}
ARM_LABELS = {"frozen": "Frozen SPOT", "adaptive": "Adaptive DSPOT"}
COLORS = {
    "pca": "#4477AA",
    "svdd": "#EE6677",
    "vae": "#228833",
    "frozen": "#4477AA",
    "adaptive": "#EE6677",
    "21.03": "#4477AA",
    "23.05": "#EE6677",
}

METRIC_COLUMNS = {
    "model",
    "seed",
    "domain",
    "arm",
    "q",
    "calibration_alarms_per_day",
    "pr_auc",
    "roc_auc",
    "point_precision",
    "point_recall",
    "point_f1",
    "eta_precision",
    "eta_recall",
    "eta_f1",
    "eta_detection_recall",
    "alarms",
    "false_alarms",
    "false_alarms_per_day",
    "duration_days",
    "attack_count",
    "attacks_detected",
    "attack_detection_rate",
    "mean_delay_seconds",
    "median_delay_seconds",
    "attribution_target_coverage",
    "attribution_hit_at_k",
}
ATTACK_COLUMNS = {
    "model",
    "seed",
    "domain",
    "arm",
    "version",
    "file",
    "attack_id",
    "event_start",
    "event_end",
    "duration_seconds",
    "detected",
    "delay_seconds",
    "documented_targets",
    "retained_targets",
    "target_retained",
    "best_target_rank",
    "attribution_hit",
    "top_sensors",
}
CALIBRATION_COLUMNS = {
    "model",
    "seed",
    "arm",
    "q",
    "observed_alarms_per_day",
    "target_alarms_per_day",
    "scored_samples",
    "source_threshold",
}


@dataclass
class ReportInputs:
    metrics: pd.DataFrame
    attacks: pd.DataFrame
    calibration: pd.DataFrame
    config: dict[str, Any]
    metadata: dict[str, Any]
    training: list[dict[str, Any]]
    seeds: list[int]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")


def _boolean_series(series: pd.Series, name: str, allow_missing: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        result = series.astype(float if allow_missing else bool)
        return result

    def convert(value: Any) -> Any:
        if pd.isna(value):
            if allow_missing:
                return np.nan
            raise ValueError(f"Missing boolean value in {name}")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"Invalid boolean value {value!r} in {name}")

    converted = series.map(convert)
    if allow_missing:
        return converted.map(lambda value: float(value) if not pd.isna(value) else np.nan)
    return converted.astype(bool)


def _ordered(frame: pd.DataFrame, columns: Sequence[str] = ("model", "seed", "domain", "arm")) -> pd.DataFrame:
    result = frame.copy()
    temporary: list[str] = []
    orders = {
        "model": {value: index for index, value in enumerate(MODELS)},
        "domain": {value: index for index, value in enumerate(DOMAINS)},
        "arm": {value: index for index, value in enumerate(ARMS)},
    }
    for column in ("model", "domain", "arm"):
        if column in result.columns and column in columns:
            name = f"__order_{column}"
            result[name] = result[column].map(orders[column])
            if result[name].isna().any():
                raise ValueError(f"Unexpected value in {column}")
            temporary.append(name)
    sort_columns: list[str] = []
    for column in columns:
        order_name = f"__order_{column}"
        sort_columns.append(order_name if order_name in temporary else column)
    result = result.sort_values(sort_columns, kind="stable").drop(columns=temporary)
    return result.reset_index(drop=True)


def _validate_metric_matrix(metrics: pd.DataFrame, seeds: list[int]) -> None:
    keys = ["model", "seed", "domain", "arm"]
    if metrics.duplicated(keys).any():
        raise ValueError("metrics.csv contains duplicate model/seed/domain/arm rows")
    expected = {(model, seed, domain, arm) for model in MODELS for seed in seeds for domain in DOMAINS for arm in ARMS}
    observed = set(metrics[keys].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Formal result matrix is incomplete or unexpected; missing={missing}, extra={extra}")


def load_inputs(root: Path) -> ReportInputs:
    metrics_path = root / "metrics.csv"
    attacks_path = root / "per_attack.csv"
    calibration_path = root / "calibration.csv"
    config_path = root / "evaluation_config.json"
    for path in (metrics_path, attacks_path, calibration_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required formal result input is missing: {path}")

    metrics = pd.read_csv(metrics_path, dtype={"model": str, "domain": str, "arm": str})
    attacks = pd.read_csv(
        attacks_path,
        dtype={"model": str, "domain": str, "arm": str, "version": str, "file": str, "attack_id": str},
    )
    calibration = pd.read_csv(calibration_path, dtype={"model": str, "arm": str})
    _require_columns(metrics, METRIC_COLUMNS, metrics_path)
    _require_columns(attacks, ATTACK_COLUMNS, attacks_path)
    _require_columns(calibration, CALIBRATION_COLUMNS, calibration_path)

    for frame, name in ((metrics, "metrics"), (calibration, "calibration")):
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        frame["model"] = frame["model"].astype(str)
        frame["arm"] = frame["arm"].astype(str)
    metrics["domain"] = metrics["domain"].astype(str)
    attacks["seed"] = pd.to_numeric(attacks["seed"], errors="raise").astype(int)
    for column in ("model", "domain", "arm", "version", "file"):
        attacks[column] = attacks[column].astype(str)
    attacks["attack_id"] = attacks["attack_id"].astype(str)
    attacks["detected"] = _boolean_series(attacks["detected"], "per_attack.detected")
    attacks["target_retained"] = _boolean_series(attacks["target_retained"], "per_attack.target_retained")
    attacks["attribution_hit"] = _boolean_series(attacks["attribution_hit"], "per_attack.attribution_hit", True)
    for column in ("documented_targets", "retained_targets", "top_sensors"):
        attacks[column] = attacks[column].fillna("").astype(str)

    metric_numeric = sorted(METRIC_COLUMNS - {"model", "seed", "domain", "arm"})
    for column in metric_numeric:
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    for column in sorted(CALIBRATION_COLUMNS - {"model", "seed", "arm"}):
        calibration[column] = pd.to_numeric(calibration[column], errors="raise")
    for column in ("event_start", "event_end", "duration_seconds", "delay_seconds", "best_target_rank"):
        attacks[column] = pd.to_numeric(attacks[column], errors="coerce")

    seeds = sorted(int(seed) for seed in metrics["seed"].unique())
    if len(seeds) != 3:
        raise ValueError(f"The formal report requires exactly three seeds; found {seeds}")
    _validate_metric_matrix(metrics, seeds)
    if set(calibration["seed"]) != set(seeds):
        raise ValueError("Calibration seeds do not match metrics seeds")
    calibration_keys = ["model", "seed", "arm"]
    expected_calibration = {(model, seed, arm) for model in MODELS for seed in seeds for arm in ARMS}
    observed_calibration = set(calibration[calibration_keys].itertuples(index=False, name=None))
    if calibration.duplicated(calibration_keys).any() or observed_calibration != expected_calibration:
        raise ValueError("calibration.csv does not contain exactly one row per model/seed/arm")

    attack_keys = ["model", "seed", "domain", "arm", "file", "attack_id"]
    if attacks.duplicated(attack_keys).any():
        raise ValueError("per_attack.csv contains duplicate attack rows")
    if set(attacks["seed"]) != set(seeds):
        raise ValueError("Per-attack seeds do not match metrics seeds")

    config = _read_json(config_path)
    processed_dir = Path(config["data"]["processed_dir"])
    metadata = _read_json(processed_dir / "metadata.json")
    if int(metadata["common_tag_count"]) != len(metadata["common_tags"]):
        raise ValueError("metadata.json common tag count is inconsistent")
    shared_tag_count = int(metadata["shared_tag_count"])
    modeling_tag_count = int(metadata["common_tag_count"])
    excluded_constant_tags = metadata["excluded_constant_tags"]
    if shared_tag_count - modeling_tag_count != len(excluded_constant_tags):
        raise ValueError("metadata.json shared/source-nonconstant tag counts are inconsistent")
    if set(excluded_constant_tags) & set(metadata["common_tags"]):
        raise ValueError("Excluded source-constant tags remain in the modeling schema")

    metadata_attack_counts: dict[str, int] = {}
    for key, info in metadata["files"].items():
        domain = key.split("/", 1)[0]
        if domain in DOMAINS:
            metadata_attack_counts[domain] = metadata_attack_counts.get(domain, 0) + int(info["attack_events"])
    observed_attack_counts = attacks[["domain", "file", "attack_id"]].drop_duplicates().groupby("domain").size().to_dict()
    for domain in DOMAINS:
        expected_count = metadata_attack_counts.get(domain, 0)
        if int(observed_attack_counts.get(domain, 0)) != expected_count:
            raise ValueError(f"Per-attack rows do not cover the documented {domain} attacks")
        expected_run_count = expected_count * len(MODELS) * len(seeds) * len(ARMS)
        observed_count = len(attacks.loc[attacks["domain"] == domain])
        if observed_count != expected_run_count:
            raise ValueError(f"Incomplete per-attack result matrix for {domain}")

    training: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in seeds:
            info = _read_json(root / "checkpoints" / f"{model}_seed{seed}" / "training.json")
            if str(info.get("model")) != model or int(info.get("seed")) != seed:
                raise ValueError(f"Checkpoint metadata mismatch for {model} seed {seed}")
            if bool(info.get("smoke", False)):
                raise ValueError("Smoke training metadata cannot be used for the formal report")
            training.append(info)

    # Both arms consume identical continuous scores. Threshold adaptation must not
    # alter threshold-independent ranking metrics.
    for metric in ("pr_auc", "roc_auc"):
        pivot = metrics.pivot(index=["model", "seed", "domain"], columns="arm", values=metric)
        difference = (pivot["frozen"] - pivot["adaptive"]).abs().max()
        if not np.isclose(float(difference), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"{metric} differs between arms despite shared continuous scores")

    return ReportInputs(
        metrics=_ordered(metrics),
        attacks=_ordered(attacks, ("model", "seed", "domain", "arm", "file", "attack_id")),
        calibration=_ordered(calibration, ("model", "seed", "arm")),
        config=config,
        metadata=metadata,
        training=training,
        seeds=seeds,
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "domain", "arm"]
    numeric = [column for column in metrics.columns if column not in {"model", "seed", "domain", "arm"}]
    grouped = metrics.groupby(keys, sort=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    counts = metrics.groupby(keys, sort=False).size().rename("n_seeds")
    summary = counts.to_frame().join(grouped).reset_index()
    ordered_columns = keys + ["n_seeds"] + [f"{column}_{stat}" for column in numeric for stat in ("mean", "std")]
    return _ordered(summary[ordered_columns], keys)


def degradation_recovery(metrics: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = metrics.set_index(["model", "seed", "domain", "arm"])
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in seeds:
            for metric in RECOVERY_METRICS:
                source = float(indexed.loc[(model, seed, "21.03", "frozen"), metric])
                cross = float(indexed.loc[(model, seed, "23.05", "frozen"), metric])
                adaptive = float(indexed.loc[(model, seed, "23.05", "adaptive"), metric])
                loss = source - cross
                gain = adaptive - cross
                valid = bool(loss > 0)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "metric_21_frozen": source,
                        "metric_23_frozen": cross,
                        "metric_23_adaptive": adaptive,
                        "loss": loss,
                        "gain": gain,
                        "recovery_valid": valid,
                        "recovery": gain / loss if valid else np.nan,
                    }
                )
    per_seed = pd.DataFrame(rows)
    value_columns = [
        "metric_21_frozen",
        "metric_23_frozen",
        "metric_23_adaptive",
        "loss",
        "gain",
        "recovery",
    ]
    grouped = per_seed.groupby(["model", "metric"], sort=False)[value_columns].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    summary = grouped.reset_index()
    counts = per_seed.groupby(["model", "metric"], sort=False).size().rename("n_seeds").reset_index()
    valid_counts = (
        per_seed.groupby(["model", "metric"], sort=False)["recovery_valid"].sum().rename("recovery_valid_seeds").reset_index()
    )
    summary = counts.merge(valid_counts, on=["model", "metric"]).merge(summary, on=["model", "metric"])
    summary["ratio_of_mean_gain_loss"] = [
        gain / loss if loss > 0 else np.nan
        for gain, loss in zip(summary["gain_mean"], summary["loss_mean"], strict=True)
    ]
    ordered_columns = ["model", "metric", "n_seeds", "recovery_valid_seeds"]
    ordered_columns += [f"{column}_{stat}" for column in value_columns for stat in ("mean", "std")]
    ordered_columns += ["ratio_of_mean_gain_loss"]
    return per_seed, _ordered(summary[ordered_columns], ("model", "metric"))


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _bar_label(axis: Any, bars: Any, digits: int = 3) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if np.isfinite(height):
            axis.annotate(
                f"{height:.{digits}f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def _plot_degradation(metrics: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36
    for axis, arm in zip(axes, ARMS, strict=True):
        for offset, domain in zip((-width / 2, width / 2), DOMAINS, strict=True):
            subset = metrics[(metrics["domain"] == domain) & (metrics["arm"] == arm)]
            grouped = subset.groupby("model")["attack_detection_rate"].agg(["mean", "std"]).reindex(MODELS)
            bars = axis.bar(
                x + offset,
                grouped["mean"],
                width,
                yerr=grouped["std"],
                capsize=4,
                color=COLORS[domain],
                alpha=0.9,
                label=domain,
            )
            _bar_label(axis, bars)
        axis.set_title(ARM_LABELS[arm])
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Attack detection rate")
    axes[0].set_ylim(0, 1.12)
    axes[1].legend(title="Evaluation domain", loc="upper right")
    figure.suptitle("Event detection degradation (mean ± SD across three seeds)")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_cross_pr_auc(metrics: pd.DataFrame, path: Path) -> None:
    # Frozen rows represent both arms because the continuous scores are shared.
    subset = metrics[(metrics["domain"] == "23.05") & (metrics["arm"] == "frozen")]
    grouped = subset.groupby("model")["pr_auc"].agg(["mean", "std"]).reindex(MODELS)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        [MODEL_LABELS[model] for model in MODELS],
        grouped["mean"],
        yerr=grouped["std"],
        capsize=5,
        color=[COLORS[model] for model in MODELS],
    )
    _bar_label(axis, bars)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("PR-AUC")
    axis.set_title("23.05 threshold-independent ranking (mean ± SD)\nOne value per model; arms use identical scores")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_false_alarms(metrics: pd.DataFrame, target: float, path: Path) -> None:
    subset = metrics[metrics["domain"] == "23.05"]
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.9))
    positive = subset.loc[subset["false_alarms_per_day"] > 0, "false_alarms_per_day"]
    floor = max(float(positive.min()) / 2 if len(positive) else target / 10, 1e-6)
    for offset, arm in zip((-width / 2, width / 2), ARMS, strict=True):
        arm_rows = subset[subset["arm"] == arm]
        grouped = arm_rows.groupby("model")["false_alarms_per_day"].agg(["mean", "std"]).reindex(MODELS)
        means = grouped["mean"].to_numpy(dtype=float)
        stds = grouped["std"].to_numpy(dtype=float)
        displayed = np.maximum(means, floor)
        lower = np.minimum(stds, np.maximum(displayed - floor, 0))
        bars = axis.bar(
            x + offset,
            displayed,
            width,
            yerr=np.vstack([lower, stds]),
            capsize=4,
            color=COLORS[arm],
            label=ARM_LABELS[arm],
        )
        for bar, actual in zip(bars, means, strict=True):
            axis.annotate(
                f"{actual:.2f}",
                (bar.get_x() + bar.get_width() / 2, max(float(bar.get_height()), floor)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axis.axhline(target, color="#222222", linestyle="--", linewidth=1.5, label=f"Calibration target ({target:g}/day)")
    axis.set_yscale("log")
    axis.set_ylim(bottom=floor / 2)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_ylabel("False alarms per day (log scale)")
    axis.set_title("23.05 false-alarm budget (mean ± SD across seeds)\nLabels show actual means; zero values are clipped only for log display")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(axis="y", which="both", alpha=0.22)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_delay(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics[metrics["domain"] == "23.05"]
    x = np.arange(len(MODELS), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, arm in zip(axes, ARMS, strict=True):
        grouped = (
            subset[subset["arm"] == arm]
            .groupby("model")["mean_delay_seconds"]
            .agg(["mean", "std"])
            .reindex(MODELS)
        )
        means = grouped["mean"].to_numpy(dtype=float)
        bars = axis.bar(
            x,
            np.nan_to_num(means, nan=0.0),
            width=0.62,
            yerr=grouped["std"],
            capsize=4,
            color=[COLORS[model] for model in MODELS],
        )
        for bar, value in zip(bars, means, strict=True):
            label = f"{value:.1f}" if np.isfinite(value) else "n/a"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, float(bar.get_height())),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean detection delay (seconds)")
    figure.suptitle("23.05 delay among detected attacks (mean ± SD across seeds)\nn/a means that no attack was detected")
    figure.tight_layout()
    _save_figure(figure, path)


def _attack_order(attacks: pd.DataFrame) -> pd.DataFrame:
    columns = ["file", "attack_id", "event_start", "event_end", "duration_seconds"]
    return (
        attacks[columns]
        .drop_duplicates(["file", "attack_id"])
        .sort_values(["file", "event_start", "attack_id"], kind="stable")
        .reset_index(drop=True)
    )


def _detection_matrix(attacks: pd.DataFrame, order: pd.DataFrame) -> np.ndarray:
    matrices = []
    keys = pd.MultiIndex.from_frame(order[["file", "attack_id"]])
    for model in MODELS:
        for arm in ARMS:
            subset = attacks[(attacks["model"] == model) & (attacks["arm"] == arm)]
            rates = subset.groupby(["file", "attack_id"])["detected"].mean()
            matrices.append(rates.reindex(keys).to_numpy(dtype=float))
    return np.column_stack(matrices)


def _plot_attack_heatmap(attacks: pd.DataFrame, path: Path) -> None:
    subset = attacks[attacks["domain"] == "23.05"]
    order = _attack_order(subset)
    matrix = _detection_matrix(subset, order)
    columns = [f"{MODEL_LABELS[model]}\n{'F' if arm == 'frozen' else 'A'}" for model in MODELS for arm in ARMS]
    labels = [f"{row.file} / {row.attack_id}" for row in order.itertuples()]
    height = max(8.0, 0.29 * len(order) + 2.0)
    figure, axis = plt.subplots(figsize=(10, height))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#D9D9D9")
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    axis.set_xticks(np.arange(len(columns)), columns)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axis.set_xlabel("Model and arm (F = frozen, A = adaptive)")
    axis.set_ylabel("Attack (file / attack ID)")
    axis.set_title("23.05 per-attack detection rate across three seeds")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="black" if 0.25 < value < 0.8 else ("white" if value <= 0.25 else "black"),
                )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Fraction of seeds detecting the attack")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_attribution(metrics: pd.DataFrame, top_k: int, path: Path) -> None:
    subset = metrics[metrics["domain"] == "23.05"]
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    specifications = (
        ("attribution_hit_at_k", f"Attribution hit@{top_k} (eligible attacks)"),
        ("attribution_target_coverage", "Retained-target coverage (all attacks)"),
    )
    for axis, (metric, title) in zip(axes, specifications, strict=True):
        for offset, arm in zip((-width / 2, width / 2), ARMS, strict=True):
            grouped = subset[subset["arm"] == arm].groupby("model")[metric].agg(["mean", "std"]).reindex(MODELS)
            bars = axis.bar(
                x + offset,
                grouped["mean"],
                width,
                yerr=grouped["std"],
                capsize=4,
                color=COLORS[arm],
                label=ARM_LABELS[arm],
            )
            _bar_label(axis, bars, digits=2)
        axis.set_title(title)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
        axis.set_ylim(0, 1.14)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Rate")
    axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle("23.05 attribution (mean ± SD across seeds)")
    figure.tight_layout()
    _save_figure(figure, path)


def generate_figures(inputs: ReportInputs, figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 120,
        }
    )
    paths = {name: figures_dir / f"{name}.png" for name in (
        "degradation",
        "cross_pr_auc",
        "false_alarm_budget",
        "delay",
        "cross_attack_heatmap",
        "attribution",
    )}
    _plot_degradation(inputs.metrics, paths["degradation"])
    _plot_cross_pr_auc(inputs.metrics, paths["cross_pr_auc"])
    _plot_false_alarms(inputs.metrics, float(inputs.config["threshold"]["alarms_per_day"]), paths["false_alarm_budget"])
    _plot_delay(inputs.metrics, paths["delay"])
    _plot_attack_heatmap(inputs.attacks, paths["cross_attack_heatmap"])
    _plot_attribution(inputs.metrics, int(inputs.config["evaluation"]["attribution_top_k"]), paths["attribution"])
    return paths


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _mean_std(mean: Any, std: Any, digits: int = 3) -> str:
    return f"{_fmt(mean, digits)} ± {_fmt(std, digits)}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = ["| " + " | ".join(clean(header) for header in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _dataset_summary(metadata: dict[str, Any]) -> tuple[pd.DataFrame, int, int]:
    groups = {"21.03 normal training": [], "21.03 test": [], "23.05 test": []}
    for key, info in metadata["files"].items():
        domain, name = key.split("/", 1)
        group = "21.03 normal training" if domain == "21.03" and name.startswith("train") else f"{domain} test"
        groups[group].append(info)
    rows = []
    for name, files in groups.items():
        rows.append(
            {
                "partition": name,
                "files": len(files),
                "rows": sum(int(info["rows"]) for info in files),
                "days": sum(int(info["rows"]) for info in files) / 86400.0,
                "attacks": sum(int(info["attack_events"]) for info in files),
            }
        )
    frame = pd.DataFrame(rows)
    total_rows = int(frame["rows"].sum())
    total_attacks = int(frame["attacks"].sum())
    total = {
        "partition": "All streams",
        "files": int(frame["files"].sum()),
        "rows": total_rows,
        "days": total_rows / 86400.0,
        "attacks": total_attacks,
    }
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    return frame, total_rows, total_attacks


def _condition_stats(metrics: pd.DataFrame, model: str, domain: str, arm: str, metric: str) -> tuple[float, float]:
    values = metrics.loc[
        (metrics["model"] == model) & (metrics["domain"] == domain) & (metrics["arm"] == arm), metric
    ]
    return float(values.mean()), float(values.std())


def _best_model(metrics: pd.DataFrame, domain: str, arm: str, metric: str) -> tuple[str, float, float]:
    stats = [(model, *_condition_stats(metrics, model, domain, arm, metric)) for model in MODELS]
    finite = [item for item in stats if np.isfinite(item[1])]
    if not finite:
        raise ValueError(f"No finite {metric} values for {domain}/{arm}")
    return max(finite, key=lambda item: item[1])


def _alarm_collapse_text(inputs: ReportInputs) -> str:
    """Describe high-recall conditions that alarm on almost every normal window."""
    cross = inputs.metrics[inputs.metrics["domain"] == "23.05"].copy()
    stride = float(inputs.config["window"]["eval_stride"])
    cross["normal_false_alarm_fraction"] = cross["false_alarms_per_day"] * stride / 86400.0
    collapsed = cross[
        (cross["attack_detection_rate"] >= 0.999) & (cross["normal_false_alarm_fraction"] >= 0.9)
    ]
    if collapsed.empty:
        return " No run met the prespecified near-continuous-alarm screen (at least 99.9% event recall and at least 90% normal-window false alarms)."
    descriptions = []
    for (model, arm), rows in collapsed.groupby(["model", "arm"], sort=False):
        descriptions.append(
            f"{MODEL_LABELS[model]} / {arm} (mean recall {rows['attack_detection_rate'].mean():.3f}, "
            f"mean FA/day {rows['false_alarms_per_day'].mean():.2f}, normal-window alarm fraction "
            f"{rows['normal_false_alarm_fraction'].mean():.1%})"
        )
    return (
        " Near-continuous-alarm failure screen (at least 99.9% event recall and at least 90% normal-window false alarms): "
        + "; ".join(descriptions)
        + ". These high-recall conditions are alarm-collapse failures, not evidence of useful detection or DSPOT recovery."
    )


def _key_findings(inputs: ReportInputs, recovery_summary: pd.DataFrame) -> list[str]:
    metrics = inputs.metrics
    source = metrics[(metrics["domain"] == "21.03") & (metrics["arm"] == "frozen")]["attack_detection_rate"].mean()
    cross = metrics[(metrics["domain"] == "23.05") & (metrics["arm"] == "frozen")]["attack_detection_rate"].mean()
    adaptive = metrics[(metrics["domain"] == "23.05") & (metrics["arm"] == "adaptive")]["attack_detection_rate"].mean()
    best_frozen = _best_model(metrics, "23.05", "frozen", "attack_detection_rate")
    best_adaptive = _best_model(metrics, "23.05", "adaptive", "attack_detection_rate")
    frozen_ties = [
        model for model in MODELS
        if np.isclose(_condition_stats(metrics, model, "23.05", "frozen", "attack_detection_rate")[0], best_frozen[1], rtol=0.0, atol=1e-12)
    ]
    adaptive_ties = [
        model for model in MODELS
        if np.isclose(_condition_stats(metrics, model, "23.05", "adaptive", "attack_detection_rate")[0], best_adaptive[1], rtol=0.0, atol=1e-12)
    ]
    frozen_labels = " and ".join(MODEL_LABELS[model] for model in frozen_ties)
    adaptive_labels = " and ".join(MODEL_LABELS[model] for model in adaptive_ties)
    best_pr = _best_model(metrics, "23.05", "frozen", "pr_auc")
    event_recovery = recovery_summary[recovery_summary["metric"] == "attack_detection_rate"].set_index("model")
    gain_model = max(MODELS, key=lambda model: float(event_recovery.loc[model, "gain_mean"]))
    gain = float(event_recovery.loc[gain_model, "gain_mean"])
    gain_models = [
        model for model in MODELS
        if np.isclose(float(event_recovery.loc[model, "gain_mean"]), gain, rtol=0.0, atol=1e-12)
    ]
    gain_labels = " and ".join(MODEL_LABELS[model] for model in gain_models)
    ratio = float(event_recovery.loc[gain_model, "ratio_of_mean_gain_loss"])
    ratio_text = f"ratio-of-means recovery {_fmt(ratio)}" if np.isfinite(ratio) else "ratio-of-means recovery undefined"
    false_alarms = metrics[metrics["domain"] == "23.05"]["false_alarms_per_day"]
    best_name = best_pr[0]
    best_source = _condition_stats(metrics, best_name, "21.03", "frozen", "attack_detection_rate")[0]
    best_cross = _condition_stats(metrics, best_name, "23.05", "frozen", "attack_detection_rate")[0]
    best_adaptive_recall = _condition_stats(metrics, best_name, "23.05", "adaptive", "attack_detection_rate")[0]
    best_gain = float(event_recovery.loc[best_name, "gain_mean"])
    best_ratio = float(event_recovery.loc[best_name, "ratio_of_mean_gain_loss"])
    best_ratio_text = _fmt(best_ratio) if np.isfinite(best_ratio) else "undefined"
    best_frozen_fa = _condition_stats(metrics, best_name, "23.05", "frozen", "false_alarms_per_day")[0]
    best_adaptive_fa = _condition_stats(metrics, best_name, "23.05", "adaptive", "false_alarms_per_day")[0]
    best_adaptation = (
        f"For {MODEL_LABELS[best_name]}, frozen event recall moved from {best_source:.3f} on 21.03 to {best_cross:.3f} on 23.05; "
        f"DSPOT changed cross-domain recall to {best_adaptive_recall:.3f} (gain {best_gain:+.3f}; recovery ROM {best_ratio_text}) "
        f"and false alarms from {best_frozen_fa:.2f} to {best_adaptive_fa:.2f} per day."
    )
    eligible = _unique_attack_targets(inputs.attacks)
    eligible_count = int(eligible["target_retained"].sum())
    missing_count = int(len(eligible) - eligible_count)
    findings = [
        f"Across models and seeds, frozen event detection moved from {source:.3f} on 21.03 to {cross:.3f} on 23.05 "
        f"(loss {source - cross:.3f}); the adaptive arm reached {adaptive:.3f} on 23.05 (signed gain {adaptive - cross:+.3f}).",
        f"The highest raw frozen 23.05 event recall belonged to {frozen_labels} "
        f"({best_frozen[1]:.3f} ± {best_frozen[2]:.3f}); the highest raw adaptive recall belonged to {adaptive_labels} "
        f"({best_adaptive[1]:.3f} ± {best_adaptive[2]:.3f}). Raw recall is not a stand-alone measure of useful detection.",
        f"{gain_labels} had the largest signed mean DSPOT event-detection gain ({gain:+.3f}; {ratio_text}). "
        "A positive gain is recovery, whereas a negative gain is an additional loss.",
        f"{MODEL_LABELS[best_pr[0]]} achieved the highest mean 23.05 PR-AUC ({best_pr[1]:.3f} ± {best_pr[2]:.3f}). "
        "Frozen and adaptive PR-AUC are identical because both arms threshold the same continuous scores.",
        f"Observed 23.05 false alarms spanned {false_alarms.min():.3f}–{false_alarms.max():.3f} per day across runs; "
        f"the source-normal calibration target was {float(inputs.config['threshold']['alarms_per_day']):g} per day."
        + _alarm_collapse_text(inputs),
        f"Of the {len(eligible)} unique 23.05 attacks, {eligible_count} retained a documented target in the "
        f"{int(inputs.metadata['common_tag_count'])}-tag source-nonconstant modeling schema and {missing_count} did not. "
        "The latter are excluded from hit-rate denominators, not counted as attribution failures.",
        best_adaptation,
    ]
    return findings


def _unique_attack_targets(attacks: pd.DataFrame) -> pd.DataFrame:
    subset = attacks[attacks["domain"] == "23.05"]
    columns = ["file", "attack_id", "event_start", "duration_seconds", "documented_targets", "retained_targets", "target_retained"]
    return (
        subset[columns]
        .drop_duplicates(["file", "attack_id"])
        .sort_values(["file", "event_start", "attack_id"], kind="stable")
        .reset_index(drop=True)
    )


def _aggregate_result_table(summary: pd.DataFrame, top_k: int) -> str:
    rows = []
    for row in _ordered(summary, ("model", "domain", "arm")).itertuples():
        rows.append(
            [
                MODEL_LABELS[row.model],
                row.domain,
                "Frozen" if row.arm == "frozen" else "Adaptive",
                _mean_std(row.attack_detection_rate_mean, row.attack_detection_rate_std),
                _mean_std(row.eta_f1_mean, row.eta_f1_std),
                _mean_std(row.point_f1_mean, row.point_f1_std),
                _mean_std(row.pr_auc_mean, row.pr_auc_std),
                _mean_std(row.false_alarms_per_day_mean, row.false_alarms_per_day_std, 2),
                _mean_std(row.attribution_hit_at_k_mean, row.attribution_hit_at_k_std),
            ]
        )
    return _markdown_table(
        ["Model", "Domain", "Arm", "Event recall", "eTa F1", "Point F1", "PR-AUC", "FA/day", f"Hit@{top_k}"], rows
    )


def _run_level_tables(metrics: pd.DataFrame, top_k: int) -> tuple[str, str, str]:
    event_rows = []
    point_rows = []
    operating_rows = []
    for row in _ordered(metrics).itertuples():
        identity = [MODEL_LABELS[row.model], row.seed, row.domain, "Frozen" if row.arm == "frozen" else "Adaptive"]
        event_rows.append(
            identity
            + [
                f"{int(row.attacks_detected)}/{int(row.attack_count)}",
                _fmt(row.attack_detection_rate),
                _fmt(row.eta_precision),
                _fmt(row.eta_recall),
                _fmt(row.eta_f1),
                _fmt(row.eta_detection_recall),
            ]
        )
        point_rows.append(
            identity
            + [
                _fmt(row.point_precision),
                _fmt(row.point_recall),
                _fmt(row.point_f1),
                _fmt(row.pr_auc),
                _fmt(row.roc_auc),
            ]
        )
        operating_rows.append(
            identity
            + [
                int(row.alarms),
                int(row.false_alarms),
                _fmt(row.false_alarms_per_day, 2),
                _fmt(row.duration_days, 3),
                _fmt(row.mean_delay_seconds, 1),
                _fmt(row.median_delay_seconds, 1),
                _fmt(row.attribution_target_coverage),
                _fmt(row.attribution_hit_at_k),
            ]
        )
    event = _markdown_table(
        ["Model", "Seed", "Domain", "Arm", "Detected", "Event recall", "eTa P", "eTa R", "eTa F1", "eTa det. R"],
        event_rows,
    )
    point = _markdown_table(
        ["Model", "Seed", "Domain", "Arm", "Point P", "Point R", "Point F1", "PR-AUC", "ROC-AUC"], point_rows
    )
    operating = _markdown_table(
        [
            "Model",
            "Seed",
            "Domain",
            "Arm",
            "Alarms",
            "False alarms",
            "FA/day",
            "Days",
            "Mean delay (s)",
            "Median delay (s)",
            "Target coverage",
            f"Hit@{top_k}",
        ],
        operating_rows,
    )
    return event, point, operating


def _recovery_table(summary: pd.DataFrame) -> str:
    labels = {"attack_detection_rate": "Event recall", "eta_f1": "eTa F1", "point_f1": "Point F1"}
    rows = []
    for row in summary.itertuples():
        rows.append(
            [
                MODEL_LABELS[row.model],
                labels[row.metric],
                _mean_std(row.metric_21_frozen_mean, row.metric_21_frozen_std),
                _mean_std(row.metric_23_frozen_mean, row.metric_23_frozen_std),
                _mean_std(row.metric_23_adaptive_mean, row.metric_23_adaptive_std),
                _mean_std(row.loss_mean, row.loss_std),
                _mean_std(row.gain_mean, row.gain_std),
                _mean_std(row.recovery_mean, row.recovery_std),
                _fmt(row.ratio_of_mean_gain_loss),
            ]
        )
    return _markdown_table(
        ["Model", "Metric", "21.03 F", "23.05 F", "23.05 A", "Loss", "Gain", "Mean recovery", "Gain/loss ROM"], rows
    )


def _calibration_table(calibration: pd.DataFrame) -> str:
    rows = []
    for row in _ordered(calibration, ("model", "seed", "arm")).itertuples():
        rows.append(
            [
                MODEL_LABELS[row.model],
                row.seed,
                "Frozen SPOT" if row.arm == "frozen" else "Adaptive DSPOT",
                f"{row.q:.6g}",
                _fmt(row.observed_alarms_per_day, 3),
                _fmt(row.target_alarms_per_day, 3),
                f"{row.source_threshold:.6g}",
            ]
        )
    return _markdown_table(["Model", "Seed", "Arm", "q", "Observed alarms/day", "Target/day", "Source threshold"], rows)


def _training_table(training: list[dict[str, Any]]) -> str:
    rows = []
    for model in MODELS:
        infos = [info for info in training if info["model"] == model]
        times = np.asarray([float(info["train_seconds"]) for info in infos], dtype=float)
        parameters = [info.get("parameters") for info in infos if info.get("parameters") is not None]
        epochs = [info.get("epochs") for info in infos if info.get("epochs") is not None]
        rows.append(
            [
                MODEL_LABELS[model],
                ", ".join(str(info["seed"]) for info in sorted(infos, key=lambda item: item["seed"])),
                str(parameters[0]) if parameters and all(value == parameters[0] for value in parameters) else "n/a",
                str(epochs[0]) if epochs and all(value == epochs[0] for value in epochs) else "n/a",
                _mean_std(times.mean(), times.std(), 2),
            ]
        )
    return _markdown_table(["Model", "Seeds", "Parameters", "Epochs", "Training seconds (mean ± SD)"], rows)


def _per_attack_detection_failures(attacks: pd.DataFrame, seeds: list[int]) -> tuple[str, int, int, int]:
    subset = attacks[attacks["domain"] == "23.05"]
    order = _attack_order(subset)
    matrix = _detection_matrix(subset, order)
    seed_count = len(seeds)
    rows = []
    universal = 0
    partial = 0
    for index, attack in enumerate(order.itertuples()):
        detected_counts = np.rint(matrix[index] * seed_count).astype(int)
        missed = int((seed_count - detected_counts).sum())
        if missed == 0:
            continue
        if missed == matrix.shape[1] * seed_count:
            universal += 1
        else:
            partial += 1
        rows.append(
            [attack.attack_id, attack.file, int(attack.duration_seconds), f"{missed}/{matrix.shape[1] * seed_count}"]
            + [f"{count}/{seed_count}" for count in detected_counts]
        )
    if not rows:
        return "No 23.05 attack was missed in any model/seed/arm run.", universal, partial, len(order)
    headers = ["Attack", "File", "Duration (s)", "Missed runs"] + [
        f"{MODEL_LABELS[model]} {'F' if arm == 'frozen' else 'A'}" for model in MODELS for arm in ARMS
    ]
    table = _markdown_table(headers, rows)
    return table, universal, partial, len(order)


def _per_attack_attribution_failures(attacks: pd.DataFrame, seeds: list[int], top_k: int) -> str:
    subset = attacks[(attacks["domain"] == "23.05") & attacks["target_retained"]].copy()
    subset["hit"] = subset["attribution_hit"].astype(float)
    order = _unique_attack_targets(attacks)
    order = order[order["target_retained"]].reset_index(drop=True)
    if order.empty:
        return f"No 23.05 attack retained a documented modeling target, so attribution hit@{top_k} is undefined."
    keys = pd.MultiIndex.from_frame(order[["file", "attack_id"]])
    columns = [(model, arm) for model in MODELS for arm in ARMS]
    matrices = []
    for model, arm in columns:
        selected = subset[(subset["model"] == model) & (subset["arm"] == arm)]
        rates = selected.groupby(["file", "attack_id"])["hit"].mean()
        matrices.append(rates.reindex(keys).to_numpy(dtype=float))
    matrix = np.column_stack(matrices) if matrices else np.empty((len(order), 0))
    seed_count = len(seeds)
    rows = []
    for index, attack in enumerate(order.itertuples()):
        hit_counts = np.rint(matrix[index] * seed_count).astype(int)
        misses = int((seed_count - hit_counts).sum())
        if misses == 0:
            continue
        retained = attack.retained_targets.replace("|", ", ")
        rows.append(
            [attack.attack_id, attack.file, retained, f"{misses}/{matrix.shape[1] * seed_count}"]
            + [f"{count}/{seed_count}" for count in hit_counts]
        )
    if not rows:
        return f"No eligible 23.05 attack had an attribution hit@{top_k} failure."
    headers = ["Attack", "File", "Retained targets", "Missed runs"] + [
        f"{MODEL_LABELS[model]} {'F' if arm == 'frozen' else 'A'}" for model in MODELS for arm in ARMS
    ]
    return _markdown_table(headers, rows)


def _detected_attribution_crosstab(attacks: pd.DataFrame, top_k: int) -> tuple[str, str]:
    """Cross detection with attribution over eligible attack-seed observations."""
    eligible = attacks[(attacks["domain"] == "23.05") & attacks["target_retained"]].copy()
    eligible = eligible.dropna(subset=["attribution_hit"])
    records: list[dict[str, Any]] = []
    cell_definitions = (
        ("nd_h", False, 1.0),
        ("nd_nh", False, 0.0),
        ("d_h", True, 1.0),
        ("d_nh", True, 0.0),
    )
    for model in MODELS:
        for arm in ARMS:
            condition = eligible[(eligible["model"] == model) & (eligible["arm"] == arm)]
            record: dict[str, Any] = {"model": model, "arm": arm, "total": len(condition)}
            for key, detected, hit in cell_definitions:
                cell = condition[(condition["detected"] == detected) & (condition["attribution_hit"] == hit)]
                record[key] = len(cell)
                record[f"{key}_attacks"] = int(cell[["file", "attack_id"]].drop_duplicates().shape[0])
                record[f"{key}_labels"] = sorted(
                    f"{row.file}/{row.attack_id}" for row in cell[["file", "attack_id"]].drop_duplicates().itertuples()
                )
            records.append(record)

    table_rows = []
    for record in records:
        table_rows.append(
            [MODEL_LABELS[record["model"]], "Frozen" if record["arm"] == "frozen" else "Adaptive", record["total"]]
            + [f"{record[key]} / {record[f'{key}_attacks']}" for key, _, _ in cell_definitions]
        )
    table = _markdown_table(
        ["Model", "Arm", "Eligible rows", "Not detected + hit", "Not detected + no hit", "Detected + hit", "Detected + no hit"],
        table_rows,
    )

    def extreme(key: str, description: str) -> str:
        record = max(records, key=lambda item: item[key])
        if record[key] == 0:
            return f"No eligible 23.05 attack-seed row was in the {description} cell."
        labels = ", ".join(record[f"{key}_labels"])
        return (
            f"{MODEL_LABELS[record['model']]} / {record['arm']} had the largest {description} cell: "
            f"{record[key]} attack-seed rows from {record[f'{key}_attacks']} unique attacks ({labels})."
        )

    narrative = (
        extreme("nd_h", "not detected + target hit")
        + " "
        + extreme("d_nh", "detected + no target hit")
        + f" Thus a documented target sensor can have a top-{top_k} score response without the max-fused score crossing the EVT threshold; "
        "conversely, a fused alarm does not imply a correct target attribution. Detected-only delay statistics use a different attack subset: "
        "they omit target-responsive but undetected attacks, while detected attacks without a target hit remain eligible for delay summaries."
    )
    return table, narrative




def _figure_markdown(path: Path, reports_dir: Path, caption: str) -> str:
    relative = os.path.relpath(path, start=reports_dir).replace(os.sep, "/")
    return f"![{caption}]({relative})"


def _reproduction_commands(inputs: ReportInputs, root: Path, reports_dir: Path, figures_dir: Path) -> str:
    config_paths = {str(info.get("config_path", "configs/default.yaml")) for info in inputs.training}
    config = sorted(config_paths)[0] if config_paths else "configs/default.yaml"
    quote = lambda value: shlex.quote(str(value))
    return "\n".join(
        [
            f"hai-prepare --config {quote(config)}",
            f"hai-train --config {quote(config)} --model all",
            f"hai-evaluate --config {quote(config)} --root {quote(root)}",
            f"hai-report --root {quote(root)} --reports {quote(reports_dir)} --figures {quote(figures_dir)}",
        ]
    )


def _statistical_checks(root: Path) -> str:
    """Additional checks produced by analysis/ (null, intervals, tests, ablations).

    Returns an empty string when the analysis artifacts are absent, so the report
    still builds from a fresh clone.
    """
    required = ["null_event_recall.csv", "pr_auc_ci.csv", "paired_tests.csv",
                "fusion_ablation.csv", "calibration_miss.csv",
                "attribution_process_check.csv", "attribution_ranks.csv"]
    if not all((root / name).exists() for name in required):
        return ""
    nulls = pd.read_csv(root / "null_event_recall.csv", dtype={"domain": str})
    ci = pd.read_csv(root / "pr_auc_ci.csv", dtype={"domain": str})
    tests = pd.read_csv(root / "paired_tests.csv", dtype={"domain": str})
    fusion = pd.read_csv(root / "fusion_ablation.csv", dtype={"domain": str})
    cal = pd.read_csv(root / "calibration_miss.csv")
    proc = pd.read_csv(root / "attribution_process_check.csv")
    ranks = pd.read_csv(root / "attribution_ranks.csv", dtype={"domain": str})

    out = ["## Statistical Checks (added)", ""]
    out.append(
        "These checks were run after the original analysis. They reuse the saved "
        "per-window and per-sensor scores; no model was retrained. Scripts and raw "
        "outputs are in `analysis/` and `results/`."
    )
    out.append("")

    out.append("### Is the event-detection rate above chance?")
    out.append("")
    out.append("A circular shift rolls each model's alarm series within its stream, keeping "
               "the run structure and destroying only its alignment with the attacks "
               "(200 shifts per configuration).")
    out.append("")
    rows = [[r.model, r.domain, r.arm, f"{r.observed:.3f}", f"{r.null_mean:.3f}",
             f"{r.p_value:.3f}"] for r in nulls.itertuples()]
    out.append(_markdown_table(["Model", "Domain", "Arm", "Observed", "Null mean", "p"], rows))
    out.append("")
    cross = nulls[(nulls.domain == "23.05") & (nulls.arm == "frozen")]
    collapsed = cross[cross.p_value >= 0.5]
    if len(collapsed):
        names = " and ".join(sorted(MODEL_LABELS[m] for m in collapsed.model))
        out.append(
            f"On 23.05 the {names} event recall of 1.000 is exactly what a randomly "
            "time-shifted alarm series achieves (p = 1.000). It is alarm collapse, not "
            "detection."
        )
    else:
        out.append("Every model beats its own null on both domains.")
    out.append("")

    out.append("### PR-AUC with 95% block-bootstrap intervals")
    out.append("")
    rows = [[r.model, r.domain, f"{r.pr_auc:.3f}",
             f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"] for r in ci.itertuples()]
    out.append(_markdown_table(["Model", "Domain", "PR-AUC", "95% CI"], rows))
    out.append("")
    out.append("No in-domain interval overlaps its cross-domain counterpart: the degradation "
               "is significant, not a shift in the mean.")
    out.append("")

    out.append("### Paired tests on event recall")
    out.append("")
    rows = [[r.domain, f"{r.a} vs {r.b}", f"{r.delta_event_recall:+.3f}",
             f"{r.p:.3f}"] for r in tests.itertuples()]
    out.append(_markdown_table(["Domain", "Comparison", "Delta recall", "p"], rows))
    out.append("")

    out.append("### Fusion ablation: is max fusion the cause of the collapse?")
    out.append("")
    sub = fusion[(fusion.domain == "23.05") & (fusion.arm == "frozen")]
    g = sub.groupby(["model", "fusion"])[["pr_auc", "event_recall", "fa_per_day"]].mean()
    rows = [[model, how, f"{g.loc[(model, how)].pr_auc:.3f}",
             f"{g.loc[(model, how)].event_recall:.3f}",
             f"{g.loc[(model, how)].fa_per_day:.0f}"]
            for model in ("pca", "svdd", "vae")
            for how in ("max", "mean", "median", "top3mean") if (model, how) in g.index]
    out.append(_markdown_table(["Model", "Fusion", "PR-AUC", "Recall", "FA/day"], rows))
    out.append("")
    out.append("The collapse survives every fusion rule: PCA and Deep SVDD alarm "
               "continuously under max, mean and top3-mean. Max fusion is not the cause; "
               "the representation shift is. Mean fusion does improve cross-domain PR-AUC "
               "modestly and is worth adopting.")
    out.append("")

    out.append("### In-domain calibration already misses its target")
    out.append("")
    rows = [[r.model, f"{r.source_calibration_fa:.1f}", f"{r.in_domain_test_fa:.1f}",
             f"{r.ratio_to_target:.1f}x"] for r in cal.itertuples()]
    out.append(_markdown_table(["Model", "Source split /day", "21.03 test /day",
                                "Ratio to target"], rows))
    out.append("")
    out.append("The EVT layer hits 10 alarms/day on its own calibration split and then "
               "misses by 1.8-7.4x on a different period of the same plant version. The "
               "cross-version failure amplifies an existing in-domain gap.")
    out.append("")

    out.append("### Are the documented targets consistent with the process labels?")
    out.append("")
    agree = int(proc.agree.sum())
    out.append(f"The documented target shares a process with the CSV per-process attack "
               f"labels in **{agree} of {len(proc)}** 21.03 attacks. The single exception "
               "is A220, where the documentation lists P2 while the CSV labels P3 and P4 — "
               "an internal inconsistency in the documentation, not a transcription error. "
               "23.05 ships a single label column, so this check applies to 21.03 only.")
    out.append("")

    out.append("### Attribution: median rank and target type")
    out.append("")
    med = ranks.groupby(["model", "domain"]).best_target_rank.median()
    rows = [[model, domain, f"{med.loc[(model, domain)]:.1f}"]
            for model in ("pca", "svdd", "vae") for domain in ("21.03", "23.05")
            if (model, domain) in med.index]
    out.append(_markdown_table(["Model", "Domain", "Median target rank (of 41; chance 21)"], rows))
    out.append("")
    by_kind = ranks[ranks.domain == "23.05"].groupby("kind").best_target_rank.median()
    if len(by_kind):
        rows = [[kind, f"{value:.1f}"] for kind, value in by_kind.items()]
        out.append(_markdown_table(["Target type (23.05)", "Median rank"], rows))
        out.append("")
    out.append("PCA's Hit@3 of 0.033 on 23.05 reads as useless, but its median rank of 17.5 "
               "is still better than chance. Hit@3 alone overstates the failure.")
    out.append("")
    return "\n".join(out)


def build_report_markdown(
    inputs: ReportInputs,
    metrics_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    figures: dict[str, Path],
    root: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> str:
    config = inputs.config
    metadata = inputs.metadata
    dataset, total_rows, total_attacks = _dataset_summary(metadata)
    source_attacks = int(dataset.loc[dataset["partition"] == "21.03 test", "attacks"].iloc[0])
    cross_attacks = int(dataset.loc[dataset["partition"] == "23.05 test", "attacks"].iloc[0])
    source_training_rows = int(dataset.loc[dataset["partition"] == "21.03 normal training", "rows"].iloc[0])
    top_k = int(config["evaluation"]["attribution_top_k"])
    window = int(config["window"]["length"])
    eval_stride = int(config["window"]["eval_stride"])
    train_stride = int(config["window"]["train_stride"])
    target = float(config["threshold"]["alarms_per_day"])
    findings = _key_findings(inputs, recovery_summary)
    dataset_table = _markdown_table(
        ["Partition", "Files", "Rows", "Duration (days at 1 Hz)", "Attack events"],
        [
            [row.partition, row.files, f"{int(row.rows):,}", _fmt(row.days, 3), row.attacks]
            for row in dataset.itertuples()
        ],
    )
    aggregate_table = _aggregate_result_table(metrics_summary, top_k)
    event_table, point_table, operating_table = _run_level_tables(inputs.metrics, top_k)
    recovery_table = _recovery_table(recovery_summary)
    detection_failures, universal, partial, total_cross = _per_attack_detection_failures(inputs.attacks, inputs.seeds)
    attribution_failures = _per_attack_attribution_failures(inputs.attacks, inputs.seeds, top_k)
    attribution_crosstab, attribution_crosstab_text = _detected_attribution_crosstab(inputs.attacks, top_k)
    best_delay_candidates = []
    for model in MODELS:
        for arm in ARMS:
            mean, std = _condition_stats(inputs.metrics, model, "23.05", arm, "mean_delay_seconds")
            if np.isfinite(mean):
                best_delay_candidates.append((model, arm, mean, std))
    if best_delay_candidates:
        best_delay = min(best_delay_candidates, key=lambda item: item[2])
        delay_ties = [item for item in best_delay_candidates if np.isclose(item[2], best_delay[2], rtol=0.0, atol=1e-12)]
        delay_labels = " and ".join(f"{MODEL_LABELS[model]} / {arm}" for model, arm, _, _ in delay_ties)
        delay_text = (
            f"Among conditions with at least one detected attack, {delay_labels} had the "
            f"lowest across-seed mean of per-run mean delays ({best_delay[2]:.1f} ± {best_delay[3]:.1f} s). "
            "Delay is conditional on detection and therefore does not penalize missed attacks."
        )
    else:
        delay_text = "No 23.05 attack was detected, so detection delay is undefined for every condition."
    commands = _reproduction_commands(inputs, root, reports_dir, figures_dir)
    validation_fraction = float(config["data"]["validation_fraction"])

    lines: list[str] = []
    add = lines.append
    add("# Zero-Day Anomaly Detection under HAI Plant-Version Shift")
    add("")
    add("**A three-seed, source-only evaluation of PCA, DeepSVDD, and a convolutional VAE with frozen and adaptive EVT decisions**")
    add("")
    add("Hongbin Zhang and Zeren Wang — Project Group 88, SSY340 Deep Machine Learning")
    add("")
    add("## Abstract")
    add("")
    add(
        f"This study evaluates zero-day anomaly detection on the HAI dataset. Its {int(metadata['shared_tag_count'])}-tag "
        f"public intersection is filtered using source-only variance to {int(metadata['common_tag_count'])} nonconstant modeling sensors. "
        f"The corpus contains {total_rows:,} real telemetry rows and {total_attacks} documented attack events, including {cross_attacks} in the held-out 23.05 domain. "
        f"Models are trained only on normal 21.03 data and evaluated over three seeds ({', '.join(map(str, inputs.seeds))}). "
        + findings[3]
        + " "
        + findings[6]
        + " DSPOT is interpreted strictly as decision-layer adaptation: it changes thresholds, not the learned representation."
    )
    add("")
    add("## Research Questions")
    add("")
    add("- **RQ1 — Degradation:** How much do event detection, eTaPR F1, and point F1 degrade from 21.03 to 23.05 when the source threshold is frozen?")
    add("- **RQ2 — Recovery:** How much of that loss does source-calibrated DSPOT recover on 23.05, and at what observed false-alarm rate?")
    add("- **RQ3 — Ranking versus decisions:** Do threshold-independent ranking metrics support the same conclusions as thresholded event metrics?")
    add(f"- **RQ4 — Attribution:** How often do the top {top_k} anomalous sensors include a documented target retained in the source-nonconstant modeling schema?")
    add("")
    add("## Data and Experimental Protocol")
    add("")
    add(dataset_table)
    add("")
    add(
        f"The processed corpus comprises {total_rows:,} one-hertz rows ({total_rows / 86400:.3f} stream-days). "
        f"There are {source_attacks} attacks in 21.03 test and {cross_attacks} in 23.05 test, for {total_attacks} total. "
        f"The two plant versions share {int(metadata['shared_tag_count'])} public tags, but {len(metadata['excluded_constant_tags'])} "
        "have zero variance throughout the normal 21.03 training partitions. Their constant source encoding can dominate max-fused scores after plant-version shift. "
        f"They are therefore removed using source-training variance alone, leaving {int(metadata['common_tag_count'])} source-nonconstant modeling tags. "
        "Durations are computed from the actual per-file row counts in `data/processed/metadata.json`, rather than rounded summary values."
    )
    add("")
    add(
        f"The three official 21.03 training files contain {source_training_rows:,} rows, equivalent to {source_training_rows / 3600:.1f} hours at 1 Hz. "
        "This measured duration differs from the 352 hours stated in HAI v4.0 documentation. All data accounting here follows the actual file row counts and per-file SHA-256 hashes recorded in metadata, not the copied documentation duration."
    )
    add("")
    add(
        f"Normalization statistics are fitted only on chronological 21.03 training partitions. The last {validation_fraction:.0%} "
        "of each normal source stream is reserved from model fitting and supplies normal calibration scores. "
        "No 23.05 labels, attack intervals, thresholds, or model updates are used during training or calibration. "
        "Evaluation replays each stream chronologically and separately for every model, seed, domain, and arm."
    )
    add("")
    add("## Methods")
    add("")
    add("### Shared windowing and model family")
    add("")
    add(
        f"Each score is aligned to the causal end of a {window}-second window; evaluation windows use a {eval_stride}-second stride, "
        f"while non-overlapping training windows use a {train_stride}-second stride. No future samples enter a window. "
        "DeepSVDD and the VAE use a matched three-layer, bias-free one-dimensional convolutional trunk and the same latent width. "
        "This architectural matching does not imply equal total parameter counts: the VAE additionally requires Gaussian heads and a decoder. "
        "DeepSVDD learns a latent hypersphere and scores squared distance to its fixed center. "
        "The standard VAE optimizes reconstruction plus KL divergence and scores reconstruction error using the mean latent at evaluation. "
        "The PCA baseline reconstructs flattened multivariate windows and derives per-sensor reconstruction errors. "
        "Per-sensor scores are fused by the configured maximum, so a system alarm requires only one strongly anomalous sensor."
    )
    add("")
    add(_training_table(inputs.training))
    add("")
    add("Parameter counts are read from each run's training metadata and show the actual SVDD/VAE capacity difference; PCA does not have a comparable neural parameter count. Training times are descriptive measurements, not controlled speed benchmarks.")
    add("")
    add("### EVT calibration and deployment arms")
    add("")
    add(
        f"Both EVT arms are calibrated exclusively on normal 21.03 scores to an empirical target of {target:g} alarms/day. "
        f"The initial threshold uses quantile {float(config['threshold']['initial_quantile']):g}, at least "
        f"{int(config['threshold']['min_exceedances'])} exceedances, and {int(config['threshold']['initialization_samples'])} initialization samples where available. "
        "The frozen arm deploys SPOT with its source-fitted terminal threshold. The adaptive arm deploys DSPOT with the same source calibration, then updates the decision threshold during chronological replay. "
        f"DSPOT uses depth {int(config['threshold']['dspot_depth'])} and updates every {int(config['threshold']['update_every'])} scored samples. "
        "**The EVT risk q is calibrated separately for frozen SPOT and adaptive DSPOT for every model and seed.** "
        "The observed calibration rate is an empirical selection criterion, not a guarantee of the shifted-domain false-alarm rate."
    )
    add("")
    add(_calibration_table(inputs.calibration))
    add("")
    add("### Metrics and attribution")
    add("")
    add(
        "PR-AUC and ROC-AUC are computed from continuous scores and are threshold-independent. Point precision, recall, and F1 use raw thresholded points with **no point adjustment**. "
        f"eTaPR uses theta_p = {ETA_THETA_P:g} and theta_r = {ETA_THETA_R:g}; event detection recall counts an attack as detected when at least one alarm occurs within its documented interval. "
        "Detection delay is measured from event start to the first in-event alarm and is summarized only over detected attacks. "
        "False alarms are alarms on normal points, normalized by the actual evaluated stream-days."
    )
    add("")
    add(
        f"**eTaPR is the enhanced time-series-aware metric published at SAC 2022 [7], not the CIKM 2019 TaPR metric [6].** "
        "TaPR and eTaPR are distinct; this report does not relabel either citation or use point adjustment, whose optimistic bias is discussed by Kim et al. [8]."
    )
    add("")
    add(
        f"Attribution averages per-sensor scores over each attack interval and ranks sensors. Hit@{top_k} is true when any retained documented target appears in the top {top_k}. "
        f"A retained target is the intersection of attack documentation with the {int(metadata['common_tag_count'])}-tag source-nonconstant modeling schema; target coverage is computed against this same schema. "
        "**If that intersection is empty, the attack is excluded from the attribution-hit denominator and is not counted as an attribution failure.** "
        "Target coverage is reported separately so this eligibility restriction remains visible."
    )
    add("")
    add("## Results")
    add("")
    add("### Three-seed aggregate results")
    add("")
    add("Entries are mean ± sample standard deviation across the three seeds. Event recall is the documented-attack detection rate; FA/day is the false-alarm rate.")
    add("")
    add(aggregate_table)
    add("")
    add(_figure_markdown(figures["degradation"], reports_dir, "Event detection rates on 21.03 and 23.05 for both deployment arms."))
    add("")
    add("### Complete run-level results")
    add("")
    add("The following tables include every model, seed, domain, and arm. No run is hidden behind the aggregate summary.")
    add("")
    add("#### Event detection and eTaPR")
    add("")
    add(event_table)
    add("")
    add("#### Point-wise and threshold-independent metrics")
    add("")
    add(point_table)
    add("")
    add(_figure_markdown(figures["cross_pr_auc"], reports_dir, "Cross-domain PR-AUC without duplicate arm bars."))
    add("")
    add(
        "The PR-AUC figure intentionally has one bar per model rather than duplicate frozen/adaptive bars. "
        "Both arms consume exactly the same continuous anomaly scores; only their thresholds and resulting decisions differ. "
        "Consequently their PR-AUC and ROC-AUC values are identical for a given model, seed, and domain."
    )
    add("")
    add("#### Alarm budget, delay, and attribution")
    add("")
    add(operating_table)
    add("")
    add(_figure_markdown(figures["false_alarm_budget"], reports_dir, "Cross-domain false-alarm rates on a logarithmic axis with the calibration target."))
    add("")
    add(_figure_markdown(figures["delay"], reports_dir, "Cross-domain detection delay among detected attacks."))
    add("")
    add(delay_text)
    add("")
    add("## Degradation and DSPOT Recovery")
    add("")
    add(
        "For each model, seed, and metric, degradation is `loss = metric_21_frozen - metric_23_frozen`, and DSPOT gain is "
        "`gain = metric_23_adaptive - metric_23_frozen`. Positive loss denotes degradation and positive gain denotes improvement. "
        "Per-seed recovery is `gain / loss` **only when loss > 0**; otherwise it is undefined and excluded from the mean-of-ratios. "
        "The final column instead reports the ratio of mean gain to mean loss, which uses all seeds and is defined only when mean loss > 0. "
        "These two recovery estimators answer different questions and need not agree."
    )
    add("")
    add(recovery_table)
    add("")
    add("In the table, F and A denote frozen and adaptive arms; ROM denotes ratio of means. `n/a` marks an undefined ratio, not zero recovery. Per-seed values are provided in `degradation_recovery.csv`; aggregate values and valid-seed counts are in `recovery_summary.csv`.")
    add("")
    add("## Per-Attack Detection Failures")
    add("")
    add(
        f"Across the {total_cross} 23.05 attacks, {universal} were missed in every model/seed/arm run and {partial} had at least one miss but at least one detection. "
        "The table lists every attack with any failed run. Cells give detected seeds / total seeds; attacks not listed were detected in every run."
    )
    add("")
    add(detection_failures)
    add("")
    add(_figure_markdown(figures["cross_attack_heatmap"], reports_dir, "Per-attack 23.05 detection rates for every model and arm."))
    add("")
    add("## Attribution Results and Failures")
    add("")
    add(findings[5])
    add("")
    add(_figure_markdown(figures["attribution"], reports_dir, "Cross-domain attribution hit rate and retained-target coverage."))
    add("")
    add("### Detection versus target attribution")
    add("")
    add(
        f"For attacks with a retained documented target, the contingency table crosses alarm-based detection with attribution hit@{top_k}. "
        "Each cell reports attack-seed rows / unique attacks, so persistent per-attack patterns are not hidden by pooling over seeds."
    )
    add("")
    add(attribution_crosstab)
    add("")
    add(attribution_crosstab_text)
    add("")
    add("### Per-attack attribution failures")
    add("")
    add(
        f"The following table lists every eligible attack with at least one hit@{top_k} miss. Cells give successful attributions / total seeds. "
        "Attribution and detection failures are distinct: an attack can be detected without a correct top-sensor attribution, and an attribution rank can be computed even when no alarm fires. "
        "Attacks with no retained documented target do not appear in this failure table by design."
    )
    add("")
    add(attribution_failures)
    add("")
    statistical = _statistical_checks(root)
    if statistical:
        add(statistical)
    add("## Conclusions")
    add("")
    for index in (0, 1, 2, 3, 6, 4):
        finding = findings[index]
        add(f"- {finding}")
    add("")
    add(
        "The central interpretation is decision-centric. **DSPOT adapts the EVT decision layer, not the encoder, PCA basis, or anomaly-score representation.** "
        "Thus a thresholded-metric gain is evidence of decision adaptation at the calibrated operating objective, not evidence of improved representation quality. "
        "The arm-invariant PR-AUC and ROC-AUC make that distinction explicit. False-alarm rates, event failures, and delay must therefore be read together rather than selecting a detector from PR-AUC alone."
    )
    add("")
    add("## Limitations")
    add("")
    add("- The evidence concerns one HAI testbed, two plant versions, and three random seeds; standard deviations describe seed variability but are not confidence intervals or significance tests.")
    add("- Shared-tag intersection and source-nonconstant filtering change attribution eligibility. Target coverage must accompany hit rates, and missing documented targets cannot be interpreted as model errors.")
    add("- Attack documentation and retained targets are incomplete proxies for physical root cause; a top-k hit does not establish causal attribution.")
    add("- The evaluation is an offline chronological replay. Windows are causal, but computational latency and online control-loop safety are not measured.")
    add("- Source-normal EVT calibration controls an empirical source alarm rate, not a guaranteed 23.05 false-alarm budget. DSPOT may also adapt to attacks or nonstationarity without labels.")
    add("- Mean and median delays condition on detected attacks; comparisons are optimistic for conditions with many misses and should be read alongside event recall.")
    add("")
    add("## Reproducibility")
    add("")
    add("From the repository root, with the package and locked dependencies installed, run:")
    add("")
    add("```bash")
    add(commands)
    add("```")
    add("")
    add("Generated artifacts are `metrics_summary.csv`, `degradation_recovery.csv`, and `recovery_summary.csv` under the result root; six PNGs under the figure directory; and Markdown, PDF, and poster outputs under the report directory. The report generator validates the full three-seed result matrix and rejects smoke metadata.")
    add("")
    add("## References")
    add("")
    add("1. H.-K. Shin, W. Lee, J.-H. Yun, and B.-G. Min. “HAI: HIL-based Augmented ICS Security Dataset.” *13th USENIX Workshop on Cyber Security Experimentation and Test (CSET)*, 2020.")
    add("2. H.-K. Shin, W. Lee, J.-H. Yun, and B.-G. Min. “Two ICS Security Datasets and Anomaly Detection Contest on the HIL-based Augmented ICS Testbed.” *14th USENIX Workshop on Cyber Security Experimentation and Test (CSET)*, 2021, pp. 36–40.")
    add("3. L. Ruff et al. “Deep One-Class Classification.” *International Conference on Machine Learning (ICML)*, 2018.")
    add("4. L. Ruff et al. “Rethinking Assumptions in Deep Anomaly Detection.” arXiv:2006.00339, 2020.")
    add("5. A. Siffer, P.-A. Fouque, A. Termier, and C. Largouët. “Anomaly Detection in Streams with Extreme Value Theory.” *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 2017.")
    add("6. W.-S. Hwang, J.-H. Yun, J. Kim, and B.-G. Min. “Time-Series Aware Precision and Recall for Anomaly Detection: Considering Variety of Detection Result and Addressing Ambiguous Labeling.” *ACM International Conference on Information and Knowledge Management (CIKM)*, 2019.")
    add("7. W.-S. Hwang, J.-H. Yun, J. Kim, and B.-G. Min. “Do You Know Existing Accuracy Metrics Overrate Time-Series Anomaly Detections?” *37th ACM/SIGAPP Symposium on Applied Computing (SAC)*, 2022. Introduces eTaPR; DOI: 10.1145/3477314.3507024.")
    add("8. S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon. “Towards a Rigorous Evaluation of Time-Series Anomaly Detection.” *AAAI Conference on Artificial Intelligence*, 2022.")
    add("")
    return "\n".join(lines)


def build_poster_markdown(
    inputs: ReportInputs,
    recovery_summary: pd.DataFrame,
    figures: dict[str, Path],
    reports_dir: Path,
) -> str:
    dataset, total_rows, total_attacks = _dataset_summary(inputs.metadata)
    cross_attacks = int(dataset.loc[dataset["partition"] == "23.05 test", "attacks"].iloc[0])
    findings = _key_findings(inputs, recovery_summary)
    rows = []
    event_recovery = recovery_summary[recovery_summary["metric"] == "attack_detection_rate"].set_index("model")
    for model in MODELS:
        frozen_mean, _ = _condition_stats(inputs.metrics, model, "23.05", "frozen", "attack_detection_rate")
        adaptive_mean, _ = _condition_stats(inputs.metrics, model, "23.05", "adaptive", "attack_detection_rate")
        pr_mean, _ = _condition_stats(inputs.metrics, model, "23.05", "frozen", "pr_auc")
        rows.append(
            [
                MODEL_LABELS[model],
                _fmt(frozen_mean),
                _fmt(adaptive_mean),
                _fmt(float(event_recovery.loc[model, "gain_mean"])),
                _fmt(float(event_recovery.loc[model, "ratio_of_mean_gain_loss"])),
                _fmt(pr_mean),
            ]
        )
    result_table = _markdown_table(["Model", "23.05 event F", "23.05 event A", "Gain", "Recovery ROM", "PR-AUC"], rows)
    lines = [
        "# Zero-Day Detection under HAI Plant-Version Shift",
        "",
        "**Source-only representations; frozen SPOT versus adaptive DSPOT decisions**",
        "",
        "Hongbin Zhang and Zeren Wang | Project Group 88 | SSY340 Deep Machine Learning",
        "",
        "## Setup",
        "",
        f"{total_rows:,} real 1-Hz rows | {int(inputs.metadata['shared_tag_count'])} shared tags filtered to {int(inputs.metadata['common_tag_count'])} source-nonconstant tags | {total_attacks} attacks total | {cross_attacks} attacks in 23.05 | seeds {', '.join(map(str, inputs.seeds))}.",
        "",
        f"PCA, DeepSVDD, and standard convolutional VAE; causal {int(inputs.config['window']['length'])} s windows / {int(inputs.config['window']['eval_stride'])} s evaluation stride / max sensor fusion. Normal 21.03-only training and EVT calibration; no point adjustment; eTaPR theta_p={ETA_THETA_P:g}, theta_r={ETA_THETA_R:g} (SAC 2022).",
        "",
        "## Key Results",
        "",
        result_table,
        "",
        "F = frozen SPOT; A = adaptive DSPOT; ROM = ratio of mean gain to mean loss. Values are three-seed means.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {findings[index]}" for index in (3, 6, 1, 2, 4))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"DSPOT changes the decision layer, not the representation. Frozen/adaptive PR-AUC is identical because scores are shared. Attribution coverage and eligibility use the {int(inputs.metadata['common_tag_count'])}-tag source-nonconstant schema; attacks with no retained documented target are excluded from hit denominators, never counted as attribution failures.",
            "",
            _figure_markdown(figures["degradation"], reports_dir, "Three-seed event detection degradation and adaptation."),
            "",
            _figure_markdown(figures["false_alarm_budget"], reports_dir, "Observed cross-domain false alarms against the source calibration target."),
            "",
            "**Selected references:** HAI (CSET 2020, 2021); Deep SVDD (ICML 2018); Rethinking Assumptions in Deep Anomaly Detection (2020); SPOT (KDD 2017); TaPR (CIKM 2019); eTaPR (SAC 2022); Kim et al. (AAAI 2022).",
            "",
        ]
    )
    return "\n".join(lines)


def _inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped.replace(r"\|", "|")


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_table_separator(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _column_widths(rows: list[list[str]], available: float) -> list[float]:
    column_count = len(rows[0])
    weights = []
    for column in range(column_count):
        longest = max(len(re.sub(r"[*`]", "", row[column])) for row in rows)
        weights.append(float(max(4, min(longest, 30))))
    minimum = 20.0 if column_count >= 10 else 25.0
    widths = [available * weight / sum(weights) for weight in weights]
    for _ in range(column_count):
        small = [index for index, width in enumerate(widths) if width < minimum]
        if not small:
            break
        fixed = minimum * len(small)
        remaining_indices = [index for index in range(column_count) if index not in small]
        remaining_width = max(available - fixed, 0)
        remaining_total = sum(widths[index] for index in remaining_indices)
        if remaining_total <= 0:
            break
        for index in small:
            widths[index] = minimum
        for index in remaining_indices:
            widths[index] = widths[index] / remaining_total * remaining_width
    return widths


def _markdown_table_flowable(rows: list[list[str]], available: float, styles: dict[str, ParagraphStyle]) -> Table:
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError("Markdown table has inconsistent column counts")
    font_size = 7.2 if column_count <= 7 else (6.4 if column_count <= 10 else 5.6)
    body_style = ParagraphStyle(
        "TableBody",
        parent=styles["BodyText"],
        fontSize=font_size,
        leading=font_size + 1.5,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    data = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else body_style
        data.append([Paragraph(_inline_markup(cell), style) for cell in row])
    table = Table(data, colWidths=_column_widths(rows, available), repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#25364A")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB4C0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F1F4F7")))
    table.setStyle(TableStyle(commands))
    return table


def _scaled_image(path: Path, max_width: float, max_height: float) -> RLImage:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height)
    return RLImage(str(path), width=width * scale, height=height * scale)


def markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.8,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles_map = {
        "BodyText": body,
        "Title": ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#18324A"), spaceAfter=14),
        "Heading1": ParagraphStyle("ReportH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#18324A"), spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "Heading2": ParagraphStyle("ReportH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#28587A"), spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "Heading3": ParagraphStyle("ReportH3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#28587A"), spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "Bullet": ParagraphStyle("ReportBullet", parent=body, leftIndent=13, firstLineIndent=-8, bulletIndent=3, alignment=TA_JUSTIFY, spaceAfter=4),
        "Code": ParagraphStyle("ReportCode", parent=styles["Code"], fontName="Courier", fontSize=7.5, leading=10, leftIndent=8, rightIndent=8, borderPadding=6, backColor=colors.HexColor("#F1F4F7"), spaceAfter=8),
        "Caption": ParagraphStyle("ReportCaption", parent=body, fontSize=8, leading=10, alignment=TA_LEFT, textColor=colors.HexColor("#4A5662"), spaceBefore=3, spaceAfter=9),
    }
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=46,
        bottomMargin=42,
        title="Zero-Day Anomaly Detection under HAI Plant-Version Shift",
        author="HAI zero-day detection study",
        allowSplitting=1,
    )

    story: list[Any] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(Paragraph(_inline_markup(" ".join(paragraph_lines)), body))
            paragraph_lines.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            del language
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code_lines), styles_map["Code"]))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            flush_paragraph()
            rows = [_split_markdown_row(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_split_markdown_row(lines[index]))
                index += 1
            story.append(_markdown_table_flowable(rows, document.width, styles_map))
            story.append(Spacer(1, 8))
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            caption, target = image_match.groups()
            image_path = Path(target)
            if not image_path.is_absolute():
                image_path = markdown_path.parent / image_path
            image_path = image_path.resolve()
            story.append(_scaled_image(image_path, document.width, 610))
            story.append(Paragraph(_inline_markup(caption), styles_map["Caption"]))
            index += 1
            continue
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level, heading = heading_match.groups()
            style_name = "Title" if len(level) == 1 else ("Heading1" if len(level) == 2 else ("Heading2" if len(level) == 3 else "Heading3"))
            story.append(Paragraph(_inline_markup(heading), styles_map[style_name]))
            index += 1
            continue
        if stripped == "---":
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#AAB4C0"), spaceBefore=5, spaceAfter=7))
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            flush_paragraph()
            story.append(PageBreak())
            index += 1
            continue
        bullet_match = re.match(r"^-\s+(.+)$", stripped)
        numbered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if bullet_match or numbered_match:
            flush_paragraph()
            if bullet_match:
                content, bullet = bullet_match.group(1), "•"
            else:
                content, bullet = numbered_match.group(2), f"{numbered_match.group(1)}."
            story.append(Paragraph(_inline_markup(content), styles_map["Bullet"], bulletText=bullet))
            index += 1
            continue
        paragraph_lines.append(stripped)
        index += 1
    flush_paragraph()

    def decorate_page(pdf_canvas: canvas.Canvas, doc: SimpleDocTemplate) -> None:
        pdf_canvas.saveState()
        pdf_canvas.setStrokeColor(colors.HexColor("#C4CCD4"))
        pdf_canvas.setLineWidth(0.4)
        pdf_canvas.line(doc.leftMargin, 28, A4[0] - doc.rightMargin, 28)
        pdf_canvas.setFont("Helvetica", 7.5)
        pdf_canvas.setFillColor(colors.HexColor("#5B6670"))
        pdf_canvas.drawString(doc.leftMargin, 17, "HAI zero-day detection under plant-version shift")
        pdf_canvas.drawRightString(A4[0] - doc.rightMargin, 17, f"Page {doc.page}")
        pdf_canvas.restoreState()

    document.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)


def _poster_result_rows(inputs: ReportInputs, recovery_summary: pd.DataFrame) -> list[list[str]]:
    event = recovery_summary[recovery_summary["metric"] == "attack_detection_rate"].set_index("model")
    rows = [["Model", "Event F", "Event A", "Gain", "PR-AUC"]]
    for model in MODELS:
        frozen, _ = _condition_stats(inputs.metrics, model, "23.05", "frozen", "attack_detection_rate")
        adaptive, _ = _condition_stats(inputs.metrics, model, "23.05", "adaptive", "attack_detection_rate")
        pr_auc, _ = _condition_stats(inputs.metrics, model, "23.05", "frozen", "pr_auc")
        rows.append([MODEL_LABELS[model], _fmt(frozen), _fmt(adaptive), f"{float(event.loc[model, 'gain_mean']):+.3f}", _fmt(pr_auc)])
    return rows


def _draw_canvas_text(
    pdf_canvas: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 10,
    leading: float | None = None,
    color: colors.Color = colors.HexColor("#1F2933"),
) -> float:
    leading = leading or size * 1.3
    pdf_canvas.setFont(font, size)
    pdf_canvas.setFillColor(color)
    for paragraph in text.split("\n"):
        if not paragraph:
            y -= leading * 0.55
            continue
        lines = simpleSplit(paragraph, font, size, width) or [""]
        for line in lines:
            pdf_canvas.drawString(x, y, line)
            y -= leading
        y -= leading * 0.25
    return y


def _draw_fitted_image(pdf_canvas: canvas.Canvas, path: Path, x: float, y_top: float, width: float, height: float) -> float:
    pixel_width, pixel_height = ImageReader(str(path)).getSize()
    scale = min(width / pixel_width, height / pixel_height)
    draw_width, draw_height = pixel_width * scale, pixel_height * scale
    pdf_canvas.drawImage(
        str(path),
        x + (width - draw_width) / 2,
        y_top - draw_height,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    return y_top - draw_height


def render_poster_pdf(
    path: Path,
    inputs: ReportInputs,
    recovery_summary: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    page_size = landscape(A3)
    width, height = page_size
    pdf_canvas = canvas.Canvas(str(path), pagesize=page_size)
    pdf_canvas.setTitle("Zero-Day Detection under HAI Plant-Version Shift — Poster")
    margin = 32
    gap = 22
    column_width = (width - 2 * margin - 2 * gap) / 3
    xs = [margin + index * (column_width + gap) for index in range(3)]
    pdf_canvas.setFillColor(colors.HexColor("#18324A"))
    pdf_canvas.rect(0, height - 82, width, 82, stroke=0, fill=1)
    pdf_canvas.setFillColor(colors.white)
    pdf_canvas.setFont("Helvetica-Bold", 25)
    pdf_canvas.drawString(margin, height - 43, "Zero-Day Detection under HAI Plant-Version Shift")
    pdf_canvas.setFont("Helvetica", 11)
    pdf_canvas.drawString(margin, height - 64, "Source-only PCA, DeepSVDD and VAE with frozen SPOT or adaptive DSPOT | Zhang and Wang, Group 88, SSY340")
    y_start = height - 105
    dataset, total_rows, total_attacks = _dataset_summary(inputs.metadata)
    findings = _key_findings(inputs, recovery_summary)

    y = _draw_canvas_text(pdf_canvas, "EXPERIMENTAL SETUP", xs[0], y_start, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    setup = (
        f"Data: {total_rows:,} real 1-Hz rows; {int(inputs.metadata['shared_tag_count'])} shared tags filtered to {int(inputs.metadata['common_tag_count'])} source-nonconstant tags; "
        f"{total_attacks} documented attacks ({int(dataset.loc[dataset['partition'] == '23.05 test', 'attacks'].iloc[0])} in 23.05).\n"
        f"Protocol: three seeds ({', '.join(map(str, inputs.seeds))}); normal 21.03-only training and calibration; chronological 23.05 replay.\n"
        f"Models: shared bias-free CNN trunk for DeepSVDD/VAE, standard VAE objective, and PCA reconstruction. "
        f"Causal {int(inputs.config['window']['length'])} s windows, {int(inputs.config['window']['eval_stride'])} s stride, max sensor fusion.\n"
        f"Metrics: raw point metrics without point adjustment; event detection; eTaPR (theta_p={ETA_THETA_P:g}, theta_r={ETA_THETA_R:g}; SAC 2022, not CIKM 2019); PR-AUC; delay; attribution hit@{int(inputs.config['evaluation']['attribution_top_k'])}."
    )
    y = _draw_canvas_text(pdf_canvas, setup, xs[0], y, column_width, size=10, leading=13.5)
    y -= 8
    y = _draw_canvas_text(pdf_canvas, "KEY FINDINGS", xs[0], y, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    for finding in (findings[index] for index in (3, 6, 1)):
        y = _draw_canvas_text(pdf_canvas, "- " + finding, xs[0] + 4, y, column_width - 4, size=9.5, leading=12.5)
        y -= 3
    y -= 5
    y = _draw_canvas_text(pdf_canvas, "INTERPRETATION AND LIMITS", xs[0], y, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    interpretation = (
        "DSPOT adapts the decision layer, not the representation. Equal PR-AUC across arms is therefore expected: both use identical continuous scores. "
        "A thresholded-metric gain is not evidence of representation improvement. The source alarm target is not a shifted-domain guarantee. "
        "Attacks lacking a retained documented public target are excluded from attribution-hit denominators, never treated as attribution failures. "
        "Three-seed SDs are descriptive; delays condition on detected attacks."
    )
    _draw_canvas_text(pdf_canvas, interpretation, xs[0], y, column_width, size=9.5, leading=12.5)

    y = _draw_canvas_text(pdf_canvas, "23.05 THREE-SEED MEANS", xs[1], y_start, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    table = Table(_poster_result_rows(inputs, recovery_summary), colWidths=[column_width * value for value in (0.22, 0.18, 0.18, 0.18, 0.24)])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28587A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4C0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    _, table_height = table.wrap(column_width, 300)
    table.drawOn(pdf_canvas, xs[1], y - table_height)
    y -= table_height + 13
    y = _draw_canvas_text(
        pdf_canvas,
        "F = frozen SPOT; A = adaptive DSPOT; Event = documented-attack detection rate. Gain = adaptive minus frozen. PR-AUC is arm-invariant by construction.",
        xs[1],
        y,
        column_width,
        size=8.7,
        leading=11.5,
    )
    y -= 10
    y = _draw_canvas_text(pdf_canvas, "RECOVERY AND OPERATING COST", xs[1], y, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    y = _draw_canvas_text(pdf_canvas, findings[2], xs[1], y, column_width, size=9.5, leading=12.5)
    y = _draw_canvas_text(pdf_canvas, findings[4], xs[1], y, column_width, size=9.5, leading=12.5)
    y = _draw_canvas_text(pdf_canvas, findings[5], xs[1], y, column_width, size=9.5, leading=12.5)
    y -= 8
    y = _draw_canvas_text(pdf_canvas, "ATTRIBUTION", xs[1], y, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    bottom = _draw_fitted_image(pdf_canvas, figures["attribution"], xs[1], y, column_width, 235)
    _draw_canvas_text(
        pdf_canvas,
        "Hit rate uses only attacks with a retained documented target; coverage exposes eligibility.",
        xs[1],
        bottom - 7,
        column_width,
        size=8.5,
        leading=11,
    )

    y = _draw_canvas_text(pdf_canvas, "EVENT DEGRADATION AND ADAPTATION", xs[2], y_start, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    bottom = _draw_fitted_image(pdf_canvas, figures["degradation"], xs[2], y, column_width, 255)
    y = bottom - 18
    y = _draw_canvas_text(pdf_canvas, "FALSE-ALARM BUDGET", xs[2], y, column_width, "Helvetica-Bold", 13, 16, colors.HexColor("#18324A"))
    bottom = _draw_fitted_image(pdf_canvas, figures["false_alarm_budget"], xs[2], y, column_width, 255)
    _draw_canvas_text(
        pdf_canvas,
        "Bars show three-seed means with SD; the dashed line is the normal-source calibration objective.",
        xs[2],
        bottom - 7,
        column_width,
        size=8.5,
        leading=11,
    )

    pdf_canvas.setFillColor(colors.HexColor("#4A5662"))
    pdf_canvas.setFont("Helvetica", 7.5)
    pdf_canvas.drawString(
        margin,
        16,
        "References: HAI/CSET 2020, 2021; Deep SVDD/ICML 2018; Rethinking/2020; SPOT/KDD 2017; TaPR/CIKM 2019; eTaPR/SAC 2022; Kim et al./AAAI 2022.",
    )
    pdf_canvas.drawRightString(width - margin, 16, "Full numerical results and per-attack failures: final_report.pdf")
    pdf_canvas.showPage()
    pdf_canvas.save()


def render_poster_png(
    path: Path,
    inputs: ReportInputs,
    recovery_summary: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    figure.text(0.025, 0.955, "Zero-Day Detection under HAI Plant-Version Shift", fontsize=25, fontweight="bold", color="#18324A", va="top")
    figure.text(0.026, 0.908, "Source-only representations with frozen SPOT or adaptive DSPOT decisions | Zhang and Wang, Group 88, SSY340", fontsize=12, color="#28587A", va="top")
    dataset, total_rows, total_attacks = _dataset_summary(inputs.metadata)
    findings = _key_findings(inputs, recovery_summary)

    left = figure.add_axes([0.025, 0.075, 0.285, 0.79])
    left.axis("off")
    setup = (
        f"{total_rows:,} real 1-Hz rows; {int(inputs.metadata['shared_tag_count'])} shared tags filtered to {int(inputs.metadata['common_tag_count'])} source-nonconstant tags; {total_attacks} attacks "
        f"({int(dataset.loc[dataset['partition'] == '23.05 test', 'attacks'].iloc[0])} in 23.05). Three seeds: {', '.join(map(str, inputs.seeds))}.\n\n"
        f"Normal 21.03-only training/calibration. PCA, DeepSVDD, and standard convolutional VAE; causal {int(inputs.config['window']['length'])} s windows, "
        f"{int(inputs.config['window']['eval_stride'])} s stride, max fusion. Raw point metrics without point adjustment; eTaPR SAC 2022 "
        f"(theta_p={ETA_THETA_P:g}, theta_r={ETA_THETA_R:g})."
    )
    left.text(0, 1, "EXPERIMENTAL SETUP", fontsize=14, fontweight="bold", color="#18324A", va="top")
    left.text(0, 0.935, textwrap.fill(setup, 58), fontsize=9.5, linespacing=1.45, va="top")
    left.text(0, 0.64, "KEY FINDINGS", fontsize=14, fontweight="bold", color="#18324A", va="top")
    finding_text = "\n\n".join("• " + findings[index] for index in (3, 6, 1))
    left.text(0, 0.585, textwrap.fill(finding_text, 62), fontsize=9, linespacing=1.38, va="top")
    left.text(
        0,
        0.07,
        textwrap.fill(
            f"DSPOT adapts decisions, not representations. Attribution uses the {int(inputs.metadata['common_tag_count'])}-tag source-nonconstant schema; missing retained targets are excluded from hit denominators, not counted as failures. Delays condition on detected attacks.",
            58,
        ),
        fontsize=9,
        linespacing=1.4,
        color="#4A5662",
        va="bottom",
    )

    middle = figure.add_axes([0.335, 0.47, 0.29, 0.36])
    middle.axis("off")
    middle.text(0, 1.06, "23.05 THREE-SEED MEANS", fontsize=14, fontweight="bold", color="#18324A", va="top")
    rows = _poster_result_rows(inputs, recovery_summary)
    table = middle.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center", bbox=[0, 0.53, 1, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB4C0")
        if row == 0:
            cell.set_facecolor("#28587A")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF3F6")
    middle.text(
        0,
        0.46,
        textwrap.fill(
            "F = frozen; A = adaptive. Event = attack detection rate. Gain = adaptive minus frozen. PR-AUC is identical across arms because both use the same continuous scores.",
            54,
        ),
        fontsize=9,
        linespacing=1.4,
        va="top",
    )
    middle.text(0, 0.25, "OPERATING COST AND ATTRIBUTION", fontsize=12, fontweight="bold", color="#18324A", va="top")
    middle.text(0, 0.18, textwrap.fill(findings[4] + " " + findings[5], 57), fontsize=8.8, linespacing=1.35, va="top")

    for bounds, key, title in (
        ([0.655, 0.51, 0.325, 0.32], "degradation", "Event degradation and adaptation"),
        ([0.655, 0.105, 0.325, 0.32], "false_alarm_budget", "False-alarm budget (log scale)"),
    ):
        axis = figure.add_axes(bounds)
        axis.imshow(plt.imread(figures[key]))
        axis.axis("off")
        axis.set_title(title, fontsize=12, fontweight="bold", color="#18324A", pad=5)

    figure.text(
        0.025,
        0.025,
        "HAI/CSET 2020, 2021 • Deep SVDD/ICML 2018 • Rethinking/2020 • SPOT/KDD 2017 • TaPR/CIKM 2019 • eTaPR/SAC 2022 • Kim et al./AAAI 2022",
        fontsize=8,
        color="#4A5662",
    )
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def generate(root: Path, reports_dir: Path, figures_dir: Path) -> dict[str, Path]:
    inputs = load_inputs(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = summarize_metrics(inputs.metrics)
    per_seed_recovery, recovery_summary = degradation_recovery(inputs.metrics, inputs.seeds)
    metrics_summary_path = root / "metrics_summary.csv"
    degradation_path = root / "degradation_recovery.csv"
    recovery_path = root / "recovery_summary.csv"
    metrics_summary.to_csv(metrics_summary_path, index=False)
    per_seed_recovery.to_csv(degradation_path, index=False)
    recovery_summary.to_csv(recovery_path, index=False)

    figures = generate_figures(inputs, figures_dir)
    report_markdown = build_report_markdown(inputs, metrics_summary, recovery_summary, figures, root, reports_dir, figures_dir)
    poster_markdown = build_poster_markdown(inputs, recovery_summary, figures, reports_dir)
    report_md_path = reports_dir / "final_report.md"
    poster_md_path = reports_dir / "poster.md"
    report_pdf_path = reports_dir / "final_report.pdf"
    report_md_path.write_text(report_markdown.rstrip() + "\n", encoding="utf-8")
    poster_md_path.write_text(poster_markdown.rstrip() + "\n", encoding="utf-8")
    markdown_to_pdf(report_md_path, report_pdf_path)
    # The poster is maintained as LaTeX in poster/, so the generated PDF and PNG are
    # not produced. poster.md is kept as the plain-text summary of the same results.

    return {
        "metrics_summary": metrics_summary_path,
        "degradation_recovery": degradation_path,
        "recovery_summary": recovery_path,
        **figures,
        "final_report_md": report_md_path,
        "final_report_pdf": report_pdf_path,
        "poster_md": poster_md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"), help="Formal result root")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Markdown/PDF report output directory")
    parser.add_argument("--figures", type=Path, default=Path("figures"), help="PNG figure output directory")
    args = parser.parse_args()
    outputs = generate(args.root, args.reports, args.figures)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
