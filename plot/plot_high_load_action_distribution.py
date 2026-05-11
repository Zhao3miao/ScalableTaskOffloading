import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = [
    ("campus", "Campus scenario"),
    ("subway_station", "Subway-station scenario"),
]
UE_SCALES = [100, 150, 200]
METHODS = [
    ("blind_ippo", "Blind"),
    ("ippo", "IPPO"),
    ("att_mappo", "ATT"),
    ("cd_mappo", "CD"),
]
ACTIONS = [
    ("action_0_ratio_mean", "Local", "#e6e6e6"),
    ("action_1_ratio_mean", "ES1", "#9ecae1"),
    ("action_2_ratio_mean", "ES2", "#fdd0a2"),
    ("action_3_ratio_mean", "ES3", "#c7e9c0"),
]


def load_action_distribution(path):
    df = pd.read_csv(path)
    required = {
        "scenario_family",
        "scenario_ue",
        "method",
        *[col for col, _, _ in ACTIONS],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[
        df["scenario_family"].isin([scenario for scenario, _ in SCENARIOS])
        & df["scenario_ue"].isin(UE_SCALES)
        & df["method"].isin([method for method, _ in METHODS])
    ].copy()

    if df.empty:
        raise ValueError("No matching high-load learning-method rows were found.")

    return df


def normalized_action_values(row):
    values = np.array([float(row[col]) for col, _, _ in ACTIONS], dtype=float)
    total = values.sum()
    if total <= 0:
        return values
    return values / total * 100.0


def draw_scenario_panel(ax, df, scenario_family, title, show_ylabel):
    width = 0.62
    group_gap = 0.92
    method_gap = 0.76
    x_positions = []
    x_labels = []
    ue_centers = []

    x = 0.0
    for ue in UE_SCALES:
        group_positions = []
        for method, method_label in METHODS:
            sub = df[
                (df["scenario_family"] == scenario_family)
                & (df["scenario_ue"] == ue)
                & (df["method"] == method)
            ]
            if sub.empty:
                raise ValueError(
                    f"Missing row: scenario={scenario_family}, UE={ue}, method={method}"
                )
            row = sub.iloc[0]
            values = normalized_action_values(row)

            bottom = 0.0
            for value, (_, action_label, color) in zip(values, ACTIONS):
                ax.bar(
                    x,
                    value,
                    width=width,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.75,
                    )
                bottom += value

            group_positions.append(x)
            x_positions.append(x)
            x_labels.append(method_label)
            x += method_gap

        ue_centers.append(float(np.mean(group_positions)))
        x += group_gap

    ax.set_title(title, fontsize=12.0, fontweight="bold", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9.0)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{tick}%" for tick in range(0, 101, 20)])
    if show_ylabel:
        ax.set_ylabel("Action distribution (%)", fontsize=11, fontweight="bold")
    else:
        ax.tick_params(axis="y", labelleft=False)

    for center, ue in zip(ue_centers, UE_SCALES):
        ax.text(
            center,
            100.8,
            f"UE={ue}",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
        )

    # Separate UE groups without adding a heavy frame.
    for left, right in zip(ue_centers[:-1], ue_centers[1:]):
        ax.axvline((left + right) / 2.0, color="#cfcfcf", linewidth=0.8, zorder=0)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", width=0.8)


def plot_action_distribution(df, output_png, output_pdf):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    if output_pdf:
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for idx, (scenario_family, title) in enumerate(SCENARIOS):
        draw_scenario_panel(
            axes[idx],
            df,
            scenario_family,
            f"({chr(ord('a') + idx)}) {title}",
            show_ylabel=(idx == 0),
        )

    legend_ax = fig.add_axes([0.34, 0.040, 0.38, 0.075])
    legend_ax.axis("off")
    for idx, (_, label, color) in enumerate(ACTIONS):
        center_x = 0.14 + idx * 0.24
        legend_ax.scatter(
            center_x,
            0.68,
            marker="s",
            s=115,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            transform=legend_ax.transAxes,
        )
        legend_ax.text(
            center_x,
            0.34,
            label,
            ha="center",
            va="top",
            fontsize=10.5,
            transform=legend_ax.transAxes,
        )
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.07, right=0.99, wspace=0.06)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Table IV as high-load action distribution stacked bars."
    )
    parser.add_argument(
        "--input",
        default="aggregated_test_results.csv",
        help="Path to aggregated numeric result CSV.",
    )
    parser.add_argument(
        "--output_png",
        default="plot/outputs/high_load_action_distribution.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output_pdf",
        default="plot/outputs/high_load_action_distribution.pdf",
        help="Output PDF path. Use an empty string to skip PDF output.",
    )
    args = parser.parse_args()

    df = load_action_distribution(args.input)
    output_pdf = args.output_pdf or None
    plot_action_distribution(df, args.output_png, output_pdf)
    print(f"Saved {args.output_png}")
    if output_pdf:
        print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
