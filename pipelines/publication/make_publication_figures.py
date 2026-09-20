"""
Generate publication-quality figures for the frozen HQNN readout study.

No quantum simulation and no statistical re-analysis are performed here.
All figures are generated exclusively from the frozen statistical outputs
created by run_main_statistics.py.

Figures
-------
Figure 1:
    Accuracy across principal readout representations under noise.

Figure 2:
    Paired accuracy gains over fixed parity.

Figure 3:
    XYZ multi-basis advantage versus strong one-basis alternatives.

Figure 4:
    Decoder-capacity interaction: Random Forest minus Logistic Regression.

Each figure is exported as:
    PNG -- convenient preview / repository asset
    PDF -- vector manuscript asset
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = (
    Path("results")
    / "publication"
    / "main_readout_study"
)

STATISTICS_DIR = RESULTS_DIR / "statistics"

FIGURES_DIR = RESULTS_DIR / "figures"

DESCRIPTIVE_PATH = (
    STATISTICS_DIR
    / "descriptive_statistics.csv"
)

PAIRED_PATH = (
    STATISTICS_DIR
    / "paired_comparisons.csv"
)

DECODER_PATH = (
    STATISTICS_DIR
    / "decoder_comparisons.csv"
)

MEASUREMENT_COST_PATH = (
    STATISTICS_DIR
    / "measurement_cost_analysis.csv"
)


# =============================================================================
# OUTPUT BASENAMES
# =============================================================================

FIGURE_1_BASENAME = (
    FIGURES_DIR
    / "figure_1_representation_noise_performance"
)

FIGURE_2_BASENAME = (
    FIGURES_DIR
    / "figure_2_gain_over_parity"
)

FIGURE_3_BASENAME = (
    FIGURES_DIR
    / "figure_3_xyz_measurement_cost_tradeoff"
)

FIGURE_4_BASENAME = (
    FIGURES_DIR
    / "figure_4_decoder_capacity_effect"
)


# =============================================================================
# CONSTANTS
# =============================================================================

NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

REPRESENTATION_LABELS = {
    "R0_parity": "Fixed parity",
    "R0L_learned_parity": "Learned parity",
    "R1_single_z": "Single Z",
    "R2_all_z": "All Z",
    "R3_zeng_xyz": "XYZ",
    "R4_probabilities": "Probabilities",
    "R5_z_zz": "Z + ZZ",
    "R6_prob_z_zz": "Prob. + Z + ZZ",
    "R7_full": "Full",
}


# Principal configurations retained in Figure 1.
FIGURE_1_REPRESENTATIONS = [
    "R0L_learned_parity",
    "R2_all_z",
    "R3_zeng_xyz",
    "R5_z_zz",
    "R7_full",
]


# Representative rich readouts retained in Figure 2.
FIGURE_2_REPRESENTATIONS = [
    "R2_all_z",
    "R3_zeng_xyz",
    "R5_z_zz",
    "R7_full",
]


# Strong one-basis alternatives for the measurement-cost figure.
FIGURE_3_ALTERNATIVES = [
    "R2_all_z",
    "R5_z_zz",
    "R6_prob_z_zz",
    "R7_full",
]


# Representative configurations for the decoder-capacity interaction.
FIGURE_4_REPRESENTATIONS = [
    "R0L_learned_parity",
    "R2_all_z",
    "R3_zeng_xyz",
    "R7_full",
]


# =============================================================================
# STYLE
# =============================================================================

def configure_matplotlib() -> None:
    """
    Configure restrained manuscript-oriented plotting defaults.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def representation_label(
    representation: str,
) -> str:
    return REPRESENTATION_LABELS.get(
        representation,
        representation,
    )


def save_figure(
    fig,
    basename: Path,
) -> None:
    """
    Save both high-resolution PNG and vector PDF versions.
    """

    png_path = basename.with_suffix(
        ".png"
    )

    pdf_path = basename.with_suffix(
        ".pdf"
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=400,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )


# =============================================================================
# VALIDATION
# =============================================================================

def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required frozen statistical output "
            f"not found: {path}"
        )


def validate_inputs() -> None:
    for path in [
        DESCRIPTIVE_PATH,
        PAIRED_PATH,
        DECODER_PATH,
        MEASUREMENT_COST_PATH,
    ]:
        require_file(
            path
        )


# =============================================================================
# FIGURE 1
# =============================================================================

def make_figure_1(
    descriptive: pd.DataFrame,
) -> None:
    """
    Random-Forest accuracy across noise for principal representations.

    Error bars are 95% confidence intervals around mean accuracy.
    """

    subset = descriptive[
        (
            descriptive["metric"]
            == "accuracy"
        )
        & (
            descriptive["decoder"]
            == "random_forest"
        )
        & (
            descriptive[
                "representation"
            ].isin(
                FIGURE_1_REPRESENTATIONS
            )
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.7, 4.2)
    )

    for representation in (
        FIGURE_1_REPRESENTATIONS
    ):
        group = subset[
            subset["representation"]
            == representation
        ].sort_values(
            "noise_level"
        )

        x = group[
            "noise_level"
        ].to_numpy()

        y = group[
            "mean"
        ].to_numpy()

        lower = (
            y
            - group[
                "ci95_lower"
            ].to_numpy()
        )

        upper = (
            group[
                "ci95_upper"
            ].to_numpy()
            - y
        )

        ax.errorbar(
            x,
            y,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            marker="o",
            capsize=3,
            label=representation_label(
                representation
            ),
        )

    ax.set_xlabel(
        "Depolarizing noise level"
    )

    ax.set_ylabel(
        "Mean test accuracy"
    )

    ax.set_title(
        "Accuracy Across Readout Representations "
        "Under Depolarizing Noise"
    )

    ax.set_xticks(
        NOISE_LEVELS
    )

    ax.set_ylim(
        0.45,
        0.93,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    save_figure(
        fig,
        FIGURE_1_BASENAME,
    )

    plt.close(
        fig
    )


# =============================================================================
# FIGURE 2
# =============================================================================

def make_figure_2(
    paired: pd.DataFrame,
) -> None:
    """
    Paired RF accuracy gain of rich representations over fixed parity.

    Positive values favor the information-rich representation.
    """

    subset = paired[
        (
            paired[
                "comparison_family"
            ]
            == "rich_vs_fixed_parity"
        )
        & (
            paired["metric"]
            == "accuracy"
        )
        & (
            paired["decoder"]
            == "random_forest"
        )
        & (
            paired[
                "representation_a"
            ].isin(
                FIGURE_2_REPRESENTATIONS
            )
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.7, 4.2)
    )

    for representation in (
        FIGURE_2_REPRESENTATIONS
    ):
        group = subset[
            subset[
                "representation_a"
            ]
            == representation
        ].sort_values(
            "noise_level"
        )

        x = group[
            "noise_level"
        ].to_numpy()

        y = group[
            "mean_difference"
        ].to_numpy()

        lower = (
            y
            - group[
                "ci95_lower"
            ].to_numpy()
        )

        upper = (
            group[
                "ci95_upper"
            ].to_numpy()
            - y
        )

        ax.errorbar(
            x,
            y,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            marker="o",
            capsize=3,
            label=representation_label(
                representation
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Depolarizing noise level"
    )

    ax.set_ylabel(
        "Paired accuracy gain over fixed parity"
    )

    ax.set_title(
        "Paired Accuracy Gain Over Fixed Parity"
    )

    ax.set_xticks(
        NOISE_LEVELS
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    save_figure(
        fig,
        FIGURE_2_BASENAME,
    )

    plt.close(
        fig
    )


# =============================================================================
# FIGURE 3
# =============================================================================

def make_figure_3(
    measurement_cost: pd.DataFrame,
) -> None:
    """
    Paired RF accuracy difference between XYZ and strong one-basis
    alternatives.

    Positive values favor XYZ.

    XYZ requires three measurement bases.
    Every plotted alternative requires one basis.
    """

    subset = measurement_cost[
        (
            measurement_cost[
                "metric"
            ]
            == "accuracy"
        )
        & (
            measurement_cost[
                "decoder"
            ]
            == "random_forest"
        )
        & (
            measurement_cost[
                "alternative"
            ].isin(
                FIGURE_3_ALTERNATIVES
            )
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.7, 4.3)
    )

    for alternative in (
        FIGURE_3_ALTERNATIVES
    ):
        group = subset[
            subset[
                "alternative"
            ]
            == alternative
        ].sort_values(
            "noise_level"
        )

        x = group[
            "noise_level"
        ].to_numpy()

        y = group[
            "xyz_minus_alternative"
        ].to_numpy()

        lower = (
            y
            - group[
                "difference_ci95_lower"
            ].to_numpy()
        )

        upper = (
            group[
                "difference_ci95_upper"
            ].to_numpy()
            - y
        )

        ax.errorbar(
            x,
            y,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            marker="o",
            capsize=3,
            label=(
                "XYZ vs "
                + representation_label(
                    alternative
                )
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Depolarizing noise level"
    )

    ax.set_ylabel(
        "Paired accuracy difference "
        "(XYZ - one-basis)"
    )

    ax.set_title(
        "XYZ Advantage Versus One-Basis Readouts"
    )

    ax.set_xticks(
        NOISE_LEVELS
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.text(
        0.02,
        0.04,
        "Measurement cost: XYZ = 3 bases; "
        "alternatives = 1 basis",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
    )

    save_figure(
        fig,
        FIGURE_3_BASENAME,
    )

    plt.close(
        fig
    )


# =============================================================================
# FIGURE 4
# =============================================================================

def make_figure_4(
    decoder: pd.DataFrame,
) -> None:
    """
    Paired Random-Forest minus Logistic-Regression accuracy difference.

    Positive values favor Random Forest.
    """

    subset = decoder[
        (
            decoder["metric"]
            == "accuracy"
        )
        & (
            decoder[
                "representation"
            ].isin(
                FIGURE_4_REPRESENTATIONS
            )
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.7, 4.2)
    )

    for representation in (
        FIGURE_4_REPRESENTATIONS
    ):
        group = subset[
            subset[
                "representation"
            ]
            == representation
        ].sort_values(
            "noise_level"
        )

        x = group[
            "noise_level"
        ].to_numpy()

        y = group[
            "mean_difference"
        ].to_numpy()

        lower = (
            y
            - group[
                "ci95_lower"
            ].to_numpy()
        )

        upper = (
            group[
                "ci95_upper"
            ].to_numpy()
            - y
        )

        ax.errorbar(
            x,
            y,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            marker="o",
            capsize=3,
            label=representation_label(
                representation
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Depolarizing noise level"
    )

    ax.set_ylabel(
        "Paired accuracy difference (RF - LR)"
    )

    ax.set_title(
        "Decoder-Capacity Effect by "
        "Measurement Representation"
    )

    ax.set_xticks(
        NOISE_LEVELS
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    save_figure(
        fig,
        FIGURE_4_BASENAME,
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline() -> None:
    print(
        "=" * 80
    )

    print(
        "PUBLICATION FIGURE GENERATION | REFINED"
    )

    print(
        "=" * 80
    )

    print(
        "\nValidating frozen statistical inputs..."
    )

    validate_inputs()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive = pd.read_csv(
        DESCRIPTIVE_PATH
    )

    paired = pd.read_csv(
        PAIRED_PATH
    )

    decoder = pd.read_csv(
        DECODER_PATH
    )

    measurement_cost = pd.read_csv(
        MEASUREMENT_COST_PATH
    )

    configure_matplotlib()

    print(
        "Generating refined Figure 1..."
    )

    make_figure_1(
        descriptive
    )

    print(
        "Generating refined Figure 2..."
    )

    make_figure_2(
        paired
    )

    print(
        "Generating refined Figure 3..."
    )

    make_figure_3(
        measurement_cost
    )

    print(
        "Generating refined Figure 4..."
    )

    make_figure_4(
        decoder
    )

    expected_outputs = []

    for basename in [
        FIGURE_1_BASENAME,
        FIGURE_2_BASENAME,
        FIGURE_3_BASENAME,
        FIGURE_4_BASENAME,
    ]:
        expected_outputs.append(
            basename.with_suffix(
                ".png"
            )
        )

        expected_outputs.append(
            basename.with_suffix(
                ".pdf"
            )
        )

    for path in expected_outputs:
        if not path.exists():
            raise RuntimeError(
                "Expected figure was not created: "
                f"{path}"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REFINED PUBLICATION FIGURES COMPLETE"
    )

    print(
        "=" * 80
    )

    for path in expected_outputs:
        print(
            f"\nSaved:\n{path}"
        )

    print(
        "\nNo quantum simulation was executed."
    )

    print(
        "No statistical quantities were recomputed."
    )

    print(
        "Figures were generated exclusively from "
        "the frozen statistical outputs."
    )


if __name__ == "__main__":
    run_pipeline()
