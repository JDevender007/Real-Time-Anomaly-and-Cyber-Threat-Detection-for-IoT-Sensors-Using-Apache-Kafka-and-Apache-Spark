"""Research figures generation using matplotlib."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")


def set_style():
    _check_matplotlib()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def plot_model_comparison(results: dict[str, dict[str, float]], save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    n_models = len(models)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        bars = axes[i].bar(models, values, color=colors[:n_models], alpha=0.85, edgecolor="white", linewidth=1.2)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/model_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str],
                          title: str = "Confusion Matrix",
                          save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm_array.shape[1]),
           yticks=np.arange(cm_array.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel="True Label", xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, format(cm_array[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_array[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_ablation_results(results_df: pd.DataFrame, x_col: str, y_col: str,
                          hue_col: str | None = None,
                          title: str = "Ablation Study",
                          save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        for name, group in results_df.groupby(hue_col):
            ax.plot(group[x_col], group[y_col], "o-", label=name, markersize=6, linewidth=2)
        ax.legend()
    else:
        ax.plot(results_df[x_col], results_df[y_col], "o-", color="#2196F3",
                markersize=8, linewidth=2)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/ablation_study.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_training_time_comparison(results: dict[str, float], save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    models = list(results.keys())
    times = list(results.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models, times, color=colors[:len(models)], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison", fontweight="bold")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", ha="left", va="center", fontsize=10)
    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/training_time.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_risk_distribution(scores: list[float], labels: list[str],
                           save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336", "critical": "#9C27B0"}
    colors = [color_map.get(l.lower(), "#607D8B") for l in labels]

    ax.hist(scores, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    for level, color in color_map.items():
        level_scores = [s for s, l in zip(scores, labels) if l.lower() == level]
        if level_scores:
            ax.axvline(x=np.mean(level_scores), color=color, linestyle="--",
                       label=f"{level.title()} (μ={np.mean(level_scores):.2f})", linewidth=2)

    ax.set_xlabel("Unified Risk Score")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/risk_distribution.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_cross_dataset_generalization(results: pd.DataFrame,
                                       save_path: Path | None = None) -> Path:
    _check_matplotlib()
    set_style()

    datasets = results["dataset"].unique()
    models = results["model"].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, ds in enumerate(datasets):
        ds_data = results[results["dataset"] == ds]
        for j, model in enumerate(models):
            model_data = ds_data[ds_data["model"] == model]
            if not model_data.empty:
                f1_mean = model_data["f1"].mean()
                f1_std = model_data["f1"].std()
                axes[i].barh(model, f1_mean, xerr=f1_std, color=colors[j % len(colors)],
                             alpha=0.85, capsize=3)
        axes[i].set_title(ds.upper())
        axes[i].set_xlim(0, 1.05)

    plt.suptitle("Cross-Dataset Generalization (F1 Score)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = save_path or Path("experiments/figures/cross_dataset.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path
