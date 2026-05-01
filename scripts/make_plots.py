#!/usr/bin/env python3

import argparse
import os
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "mplconfig"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ORDER = [
    "no_context",
    "relevant_only",
    "random_1",
    "random_3",
    "random_5",
    "adversarial_1",
    "misleading_only",
    "position_first",
    "position_middle",
    "position_last",
]

LABELS = {
    "no_context": "No\ncontext",
    "relevant_only": "Relevant\nonly",
    "random_1": "Random-1",
    "random_3": "Random-3",
    "random_5": "Random-5",
    "adversarial_1": "Adversarial-1",
    "misleading_only": "Misleading\nonly",
    "position_first": "First",
    "position_middle": "Middle",
    "position_last": "Last",
}

COLORS = {
    "neutral": "#7C8C9A",
    "relevant": "#2A9D8F",
    "random": "#4E79A7",
    "adversarial": "#E76F51",
    "misleading": "#B56576",
    "position": "#E9C46A",
    "ink": "#213547",
    "grid": "#D7DEE5",
    "paper": "#F5F1E8",
    "baseline": "#6C757D",
}

CONDITION_COLORS = {
    "no_context": COLORS["neutral"],
    "relevant_only": COLORS["relevant"],
    "random_1": COLORS["random"],
    "random_3": COLORS["random"],
    "random_5": COLORS["random"],
    "adversarial_1": COLORS["adversarial"],
    "misleading_only": COLORS["misleading"],
    "position_first": COLORS["position"],
    "position_middle": COLORS["position"],
    "position_last": COLORS["position"],
}


def set_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": COLORS["paper"],
            "axes.edgecolor": "#CFD7DE",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 15,
            "legend.frameon": False,
        },
    )


def sort_conditions(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.copy()
    subset["condition"] = pd.Categorical(subset["condition"], ORDER, ordered=True)
    subset = subset.sort_values("condition")
    subset["plot_label"] = subset["condition"].astype(str).map(LABELS)
    subset["percent"] = 100 * subset["value"].astype(float)
    return subset


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def style_axis(ax: plt.Axes, ylabel: str, xlabel: str, title: str, ylim=(0, 108)) -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12)
    ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def add_bar_labels(ax: plt.Axes, values: list[float], inside_threshold: float = 96.0) -> None:
    for patch, value in zip(ax.patches, values):
        x = patch.get_x() + patch.get_width() / 2
        if value >= inside_threshold:
            y = value - 2.0
            va = "top"
            color = COLORS["ink"]
        else:
            y = value + 1.4
            va = "bottom"
            color = COLORS["ink"]
        ax.text(
            x,
            y,
            f"{value:.0f}%",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=color,
        )


def add_line_labels(ax: plt.Axes, xs: list[int], ys: list[float]) -> None:
    for x, y in zip(xs, ys):
        ax.text(
            x,
            y + 1.5,
            f"{y:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["ink"],
        )


def add_horizontal_bar_labels(ax: plt.Axes, values: list[float]) -> None:
    for patch, value in zip(ax.patches, values):
        y = patch.get_y() + patch.get_height() / 2
        if value >= 0:
            x = value + 1.2
            ha = "left"
        else:
            x = value - 1.2
            ha = "right"
        ax.text(
            x,
            y,
            f"{value:+.0f} pts",
            ha=ha,
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["ink"],
        )


def plot_accuracy_by_condition(df: pd.DataFrame, out_dir: Path) -> None:
    subset = df[["condition", "accuracy"]].rename(columns={"accuracy": "value"})
    subset = sort_conditions(subset)
    palette = {label: CONDITION_COLORS[condition] for label, condition in zip(subset["plot_label"], subset["condition"].astype(str))}

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    sns.barplot(
        data=subset,
        x="plot_label",
        y="percent",
        hue="plot_label",
        dodge=False,
        order=list(subset["plot_label"]),
        hue_order=list(subset["plot_label"]),
        palette=palette,
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    style_axis(ax, "Exact-match accuracy (%)", "Context condition", "Accuracy by Context Condition")
    ax.tick_params(axis="x", rotation=28)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    baseline = float(subset.loc[subset["condition"] == "relevant_only", "percent"].iloc[0])
    ax.axhline(
        baseline,
        color=COLORS["baseline"],
        linestyle=(0, (4, 4)),
        linewidth=1.3,
        label=f"Relevant-only baseline ({baseline:.0f}%)",
    )
    ax.legend(loc="upper right")
    add_bar_labels(ax, subset["percent"].tolist())
    savefig(fig, out_dir / "accuracy_by_condition.png")


def plot_distractor_count(df: pd.DataFrame, out_dir: Path) -> None:
    cond_to_count = {"relevant_only": 0, "random_1": 1, "random_3": 3, "random_5": 5}
    subset = df[df["condition"].isin(cond_to_count)][["condition", "accuracy"]].copy()
    subset["distractor_count"] = subset["condition"].map(cond_to_count)
    subset["percent"] = 100 * subset["accuracy"].astype(float)
    subset = subset.sort_values("distractor_count")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.lineplot(
        data=subset,
        x="distractor_count",
        y="percent",
        marker="o",
        markersize=8,
        linewidth=2.8,
        color=COLORS["random"],
        ax=ax,
    )
    ax.fill_between(
        subset["distractor_count"].tolist(),
        subset["percent"].tolist(),
        [0] * len(subset),
        color=COLORS["random"],
        alpha=0.10,
    )
    style_axis(ax, "Exact-match accuracy (%)", "Number of random distractors", "Effect of Random Distractors")
    ax.set_xticks([0, 1, 3, 5])
    add_line_labels(ax, subset["distractor_count"].tolist(), subset["percent"].tolist())

    delta = subset["percent"].iloc[-1] - subset["percent"].iloc[0]
    ax.text(
        0.03,
        0.92,
        f"Net change: {delta:+.1f} pts",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#E1E7EC"},
    )
    savefig(fig, out_dir / "distractor_count_curve.png")


def plot_position_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["position_first", "position_middle", "position_last"]
    subset = df[df["condition"].isin(order)][["condition", "accuracy"]].rename(columns={"accuracy": "value"})
    subset["condition"] = pd.Categorical(subset["condition"], order, ordered=True)
    subset = subset.sort_values("condition")
    subset["plot_label"] = subset["condition"].astype(str).map(LABELS)
    subset["percent"] = 100 * subset["value"].astype(float)

    colors = [COLORS["position"]] * len(subset)
    best_index = subset["percent"].idxmax()
    colors[list(subset.index).index(best_index)] = "#D5A52B"
    palette = {label: color for label, color in zip(subset["plot_label"], colors)}

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    sns.barplot(
        data=subset,
        x="plot_label",
        y="percent",
        hue="plot_label",
        dodge=False,
        order=list(subset["plot_label"]),
        hue_order=list(subset["plot_label"]),
        palette=palette,
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    style_axis(ax, "Exact-match accuracy (%)", "Relevant passage position", "Evidence Position Sensitivity")
    add_bar_labels(ax, subset["percent"].tolist())
    savefig(fig, out_dir / "position_sensitivity.png")


def plot_evidence_attribution(df: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "relevant_only",
        "random_1",
        "random_3",
        "random_5",
        "adversarial_1",
        "position_first",
        "position_middle",
        "position_last",
    ]
    subset = (
        df[df["condition"].isin(order)][["condition", "evidence_attribution_accuracy"]]
        .dropna()
        .rename(columns={"evidence_attribution_accuracy": "value"})
    )
    subset["condition"] = pd.Categorical(subset["condition"], order, ordered=True)
    subset = subset.sort_values("condition")
    subset["plot_label"] = subset["condition"].astype(str).map(LABELS)
    subset["percent"] = 100 * subset["value"].astype(float)
    palette = {label: CONDITION_COLORS[condition] for label, condition in zip(subset["plot_label"], subset["condition"].astype(str))}

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    sns.barplot(
        data=subset,
        x="plot_label",
        y="percent",
        hue="plot_label",
        dodge=False,
        order=list(subset["plot_label"]),
        hue_order=list(subset["plot_label"]),
        palette=palette,
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    style_axis(ax, "Evidence attribution accuracy (%)", "Context condition", "Evidence Attribution by Condition")
    ax.tick_params(axis="x", rotation=28)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    add_bar_labels(ax, subset["percent"].tolist(), inside_threshold=97.0)
    ax.text(
        0.985,
        0.06,
        "Adversarial-1 is the main attribution failure point.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#E1E7EC"},
    )
    savefig(fig, out_dir / "evidence_attribution.png")


def plot_distractor_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    subset = (
        df[["condition", "distractor_sensitivity"]]
        .dropna()
        .rename(columns={"distractor_sensitivity": "value"})
    )
    subset = sort_conditions(subset)
    subset["percent"] = 100 * subset["value"].astype(float)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.barplot(
        data=subset,
        y="plot_label",
        x="percent",
        hue="plot_label",
        dodge=False,
        order=list(reversed(subset["plot_label"].tolist())),
        hue_order=list(reversed(subset["plot_label"].tolist())),
        palette={label: CONDITION_COLORS[condition] for label, condition in zip(subset["plot_label"], subset["condition"].astype(str))},
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    ax.set_title("Distractor Sensitivity by Condition", pad=12)
    ax.set_xlabel("Accuracy drop relative to relevant-only (points)")
    ax.set_ylabel("Context condition")
    ax.axvline(0, color=COLORS["baseline"], linestyle=(0, (4, 4)), linewidth=1.3)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    spread = max(abs(subset["percent"].min()), abs(subset["percent"].max()))
    ax.set_xlim(-max(8, spread + 5), max(8, spread + 8))
    add_horizontal_bar_labels(ax, list(reversed(subset["percent"].tolist())))
    savefig(fig, out_dir / "dss_by_condition.png")


def plot_misleading_context_susceptibility(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["adversarial_1", "misleading_only"]
    subset = (
        df[df["condition"].isin(order)][["condition", "misleading_context_susceptibility"]]
        .dropna()
        .rename(columns={"misleading_context_susceptibility": "value"})
    )
    subset["condition"] = pd.Categorical(subset["condition"], order, ordered=True)
    subset = subset.sort_values("condition")
    subset["plot_label"] = subset["condition"].astype(str).map(LABELS)
    subset["percent"] = 100 * subset["value"].astype(float)
    palette = {label: CONDITION_COLORS[condition] for label, condition in zip(subset["plot_label"], subset["condition"].astype(str))}

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.barplot(
        data=subset,
        x="plot_label",
        y="percent",
        hue="plot_label",
        dodge=False,
        order=list(subset["plot_label"]),
        hue_order=list(subset["plot_label"]),
        palette=palette,
        saturation=0.95,
        legend=False,
        ax=ax,
    )
    style_axis(
        ax,
        "Misleading context susceptibility (%)",
        "Condition",
        "Copying the Adversarial Answer",
        ylim=(0, 45),
    )
    add_bar_labels(ax, subset["percent"].tolist(), inside_threshold=40.0)
    ax.text(
        0.03,
        0.92,
        "Misleading-only prompts trigger far more adversarial copying.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#E1E7EC"},
    )
    savefig(fig, out_dir / "misleading_context_susceptibility.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics)
    set_theme()
    plot_accuracy_by_condition(df, out_dir)
    plot_distractor_count(df, out_dir)
    plot_position_sensitivity(df, out_dir)
    plot_evidence_attribution(df, out_dir)
    plot_distractor_sensitivity(df, out_dir)
    plot_misleading_context_susceptibility(df, out_dir)


if __name__ == "__main__":
    main()
