"""
Generate publication tables for the frozen HQNN main readout study.

This script performs:
    - no quantum simulation,
    - no model retraining,
    - no new statistical testing.

It extracts manuscript-ready tables from the frozen experimental
and statistical outputs.

Outputs
-------
Table I:
    Experimental protocol and readout representations.

Table II:
    Main Random-Forest accuracy results at selected noise levels.

Table III:
    Key paired inferential comparisons.

Each table is exported as CSV and LaTeX.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

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

STATISTICS_DIR = (
    RESULTS_DIR
    / "statistics"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

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

ENRICHED_PATH = (
    RESULTS_DIR
    / "main_readout_study_enriched.csv"
)


# =============================================================================
# OUTPUT PATHS
# =============================================================================

TABLE_1_CSV = (
    TABLES_DIR
    / "table_1_experimental_protocol.csv"
)

TABLE_1_TEX = (
    TABLES_DIR
    / "table_1_experimental_protocol.tex"
)

TABLE_2_CSV = (
    TABLES_DIR
    / "table_2_main_performance.csv"
)

TABLE_2_TEX = (
    TABLES_DIR
    / "table_2_main_performance.tex"
)

TABLE_3_CSV = (
    TABLES_DIR
    / "table_3_key_inference.csv"
)

TABLE_3_TEX = (
    TABLES_DIR
    / "table_3_key_inference.tex"
)


# =============================================================================
# LOCKED PROTOCOL
# =============================================================================

NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

SELECTED_NOISE_LEVELS = [
    0.00,
    0.10,
    0.20,
]

NUM_QUBITS = 4
SHOTS = 1024
NUM_SEEDS = 20
ARCHITECTURE = "Ring"
NOISE_MODEL = "Depolarizing"


# =============================================================================
# REPRESENTATION METADATA
# =============================================================================

REPRESENTATION_METADATA = [
    {
        "ID": "R0",
        "Representation": "Fixed parity",
        "Features": 1,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "Fixed threshold",
    },
    {
        "ID": "R0L",
        "Representation": "Learned parity",
        "Features": 1,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R1",
        "Representation": "Single Z",
        "Features": 1,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R2",
        "Representation": "All Z",
        "Features": 4,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R3",
        "Representation": "XYZ",
        "Features": 12,
        "Measurement bases": 3,
        "Basis": "X, Y, Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R4",
        "Representation": "Probabilities",
        "Features": 16,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R5",
        "Representation": "Z + ZZ",
        "Features": 10,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R6",
        "Representation": "Prob. + Z + ZZ",
        "Features": 26,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
    {
        "ID": "R7",
        "Representation": "Full",
        "Features": 31,
        "Measurement bases": 1,
        "Basis": "Z",
        "Decoder": "LR / RF",
    },
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


TABLE_2_REPRESENTATIONS = [
    "R0L_learned_parity",
    "R2_all_z",
    "R3_zeng_xyz",
    "R5_z_zz",
    "R7_full",
]


# =============================================================================
# VALIDATION
# =============================================================================

def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen output not found: {path}"
        )


def validate_inputs() -> None:
    for path in [
        DESCRIPTIVE_PATH,
        PAIRED_PATH,
        DECODER_PATH,
        MEASUREMENT_COST_PATH,
        ENRICHED_PATH,
    ]:
        require_file(path)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def representation_label(
    representation: str,
) -> str:
    return REPRESENTATION_LABELS.get(
        representation,
        representation,
    )


def format_ci(
    mean: float,
    lower: float,
    upper: float,
) -> str:
    return (
        f"{mean:.3f} "
        f"[{lower:.3f}, {upper:.3f}]"
    )


def format_difference(
    mean: float,
    lower: float,
    upper: float,
) -> str:
    return (
        f"{mean:+.3f} "
        f"[{lower:+.3f}, {upper:+.3f}]"
    )


def format_p(
    value: float,
) -> str:
    if pd.isna(value):
        return ""

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def format_effect(
    value: float,
) -> str:
    if pd.isna(value):
        return ""

    return f"{value:.2f}"


# =============================================================================
# TABLE I
# =============================================================================

def build_table_1() -> pd.DataFrame:
    """
    Representation-level experimental specification.
    """

    table = pd.DataFrame(
        REPRESENTATION_METADATA
    )

    return table


# =============================================================================
# TABLE II
# =============================================================================

def build_table_2(
    descriptive: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main RF accuracy results at clean, moderate, and severe noise.

    Values are mean accuracy with 95% confidence intervals.
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
                TABLE_2_REPRESENTATIONS
            )
        )
        & (
            descriptive[
                "noise_level"
            ].isin(
                SELECTED_NOISE_LEVELS
            )
        )
    ].copy()

    rows = []

    for representation in (
        TABLE_2_REPRESENTATIONS
    ):
        row = {
            "Representation": (
                representation_label(
                    representation
                )
            )
        }

        representation_data = subset[
            subset["representation"]
            == representation
        ]

        for noise in (
            SELECTED_NOISE_LEVELS
        ):
            match = representation_data[
                np.isclose(
                    representation_data[
                        "noise_level"
                    ],
                    noise,
                )
            ]

            if len(match) != 1:
                raise RuntimeError(
                    "Expected exactly one descriptive "
                    "result for "
                    f"{representation}, noise={noise}; "
                    f"found {len(match)}."
                )

            result = match.iloc[0]

            row[
                f"Noise {noise:.2f}"
            ] = format_ci(
                float(result["mean"]),
                float(result["ci95_lower"]),
                float(result["ci95_upper"]),
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# TABLE III HELPERS
# =============================================================================

def extract_paired_row(
    paired: pd.DataFrame,
    family: str,
    noise: float,
    representation_a: str,
    decoder: str,
) -> pd.Series:

    subset = paired[
        (
            paired[
                "comparison_family"
            ]
            == family
        )
        & (
            paired["metric"]
            == "accuracy"
        )
        & (
            paired["decoder"]
            == decoder
        )
        & (
            paired[
                "representation_a"
            ]
            == representation_a
        )
        & (
            np.isclose(
                paired[
                    "noise_level"
                ],
                noise,
            )
        )
    ]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one paired result for "
            f"{family}, {representation_a}, "
            f"{decoder}, noise={noise}; "
            f"found {len(subset)}."
        )

    return subset.iloc[0]


def extract_decoder_row(
    decoder: pd.DataFrame,
    noise: float,
    representation: str,
) -> pd.Series:

    subset = decoder[
        (
            decoder["metric"]
            == "accuracy"
        )
        & (
            decoder[
                "representation"
            ]
            == representation
        )
        & (
            np.isclose(
                decoder[
                    "noise_level"
                ],
                noise,
            )
        )
    ]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one decoder result for "
            f"{representation}, noise={noise}; "
            f"found {len(subset)}."
        )

    return subset.iloc[0]


def extract_cost_row(
    measurement_cost: pd.DataFrame,
    noise: float,
    alternative: str,
) -> pd.Series:

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
            ]
            == alternative
        )
        & (
            np.isclose(
                measurement_cost[
                    "noise_level"
                ],
                noise,
            )
        )
    ]

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one measurement-cost "
            "result for "
            f"{alternative}, noise={noise}; "
            f"found {len(subset)}."
        )

    return subset.iloc[0]


# =============================================================================
# TABLE III
# =============================================================================

def build_table_3(
    paired: pd.DataFrame,
    decoder: pd.DataFrame,
    measurement_cost: pd.DataFrame,
) -> pd.DataFrame:
    """
    Key inferential comparisons supporting the principal manuscript claims.
    """

    rows: List[dict] = []

    # -------------------------------------------------------------------------
    # Rich representation versus fixed parity
    # -------------------------------------------------------------------------

    for noise in [
        0.00,
        0.10,
        0.20,
    ]:
        result = extract_paired_row(
            paired=paired,
            family="rich_vs_fixed_parity",
            noise=noise,
            representation_a="R3_zeng_xyz",
            decoder="random_forest",
        )

        rows.append(
            {
                "Comparison": (
                    "XYZ vs fixed parity"
                ),
                "Noise": noise,
                "Paired difference [95% CI]": (
                    format_difference(
                        float(
                            result[
                                "mean_difference"
                            ]
                        ),
                        float(
                            result[
                                "ci95_lower"
                            ]
                        ),
                        float(
                            result[
                                "ci95_upper"
                            ]
                        ),
                    )
                ),
                "Cohen dz": format_effect(
                    float(
                        result[
                            "cohens_dz"
                        ]
                    )
                ),
                "Holm p": format_p(
                    float(
                        result[
                            "paired_t_p_holm"
                        ]
                    )
                ),
            }
        )

    # -------------------------------------------------------------------------
    # XYZ versus one-basis alternatives
    # -------------------------------------------------------------------------

    for alternative in [
        "R2_all_z",
        "R5_z_zz",
        "R7_full",
    ]:
        for noise in [
            0.00,
            0.10,
            0.20,
        ]:
            result = extract_cost_row(
                measurement_cost=(
                    measurement_cost
                ),
                noise=noise,
                alternative=alternative,
            )

            rows.append(
                {
                    "Comparison": (
                        "XYZ vs "
                        + representation_label(
                            alternative
                        )
                    ),
                    "Noise": noise,
                    "Paired difference [95% CI]": (
                        format_difference(
                            float(
                                result[
                                    "xyz_minus_alternative"
                                ]
                            ),
                            float(
                                result[
                                    "difference_ci95_lower"
                                ]
                            ),
                            float(
                                result[
                                    "difference_ci95_upper"
                                ]
                            ),
                        )
                    ),
                    "Cohen dz": format_effect(
                        float(
                            result[
                                "cohens_dz"
                            ]
                        )
                    ),
                    "Holm p": format_p(
                        float(
                            result[
                                "paired_t_p_holm"
                            ]
                        )
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Decoder-capacity interaction
    # -------------------------------------------------------------------------

    for representation in [
        "R0L_learned_parity",
        "R2_all_z",
        "R3_zeng_xyz",
        "R7_full",
    ]:
        for noise in [
            0.00,
            0.10,
            0.20,
        ]:
            result = extract_decoder_row(
                decoder=decoder,
                noise=noise,
                representation=representation,
            )

            rows.append(
                {
                    "Comparison": (
                        "RF - LR: "
                        + representation_label(
                            representation
                        )
                    ),
                    "Noise": noise,
                    "Paired difference [95% CI]": (
                        format_difference(
                            float(
                                result[
                                    "mean_difference"
                                ]
                            ),
                            float(
                                result[
                                    "ci95_lower"
                                ]
                            ),
                            float(
                                result[
                                    "ci95_upper"
                                ]
                            ),
                        )
                    ),
                    "Cohen dz": format_effect(
                        float(
                            result[
                                "cohens_dz"
                            ]
                        )
                    ),
                    "Holm p": format_p(
                        float(
                            result[
                                "paired_t_p_holm"
                            ]
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# LATEX EXPORT
# =============================================================================

def save_latex_table(
    table: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    """
    Export a compact LaTeX table.

    Escaping remains enabled so representation labels and symbols
    are safe for direct manuscript inclusion.
    """

    latex = table.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
    )

    path.write_text(
        latex,
        encoding="utf-8",
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_tables(
    table_1: pd.DataFrame,
    table_2: pd.DataFrame,
    table_3: pd.DataFrame,
) -> None:

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_1.to_csv(
        TABLE_1_CSV,
        index=False,
    )

    save_latex_table(
        table=table_1,
        path=TABLE_1_TEX,
        caption=(
            "Readout representations and "
            "decoder configurations."
        ),
        label=(
            "tab:readout_representations"
        ),
    )

    table_2.to_csv(
        TABLE_2_CSV,
        index=False,
    )

    save_latex_table(
        table=table_2,
        path=TABLE_2_TEX,
        caption=(
            "Random-Forest test accuracy for "
            "principal readout representations. "
            "Values are means with 95 percent "
            "confidence intervals across 20 seeds."
        ),
        label=(
            "tab:main_performance"
        ),
    )

    table_3.to_csv(
        TABLE_3_CSV,
        index=False,
    )

    save_latex_table(
        table=table_3,
        path=TABLE_3_TEX,
        caption=(
            "Selected paired inferential comparisons. "
            "Differences are computed across matched "
            "experimental seeds; p-values are "
            "Holm-adjusted."
        ),
        label=(
            "tab:key_inference"
        ),
    )


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_tables(
    table_1: pd.DataFrame,
    table_2: pd.DataFrame,
    table_3: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TABLE I | READOUT REPRESENTATIONS"
    )

    print(
        "=" * 80
    )

    print(
        table_1.to_string(
            index=False
       )

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TABLE II | MAIN PERFORMANCE RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        table_2.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TABLE III | KEY PAIRED INFERENCE"
    )

    print(
        "=" * 80
    )

    print(
        table_3.to_string(
            index=False
        )
    )


# =============================================================================
# PROTOCOL REPORT
# =============================================================================

def print_protocol() -> None:
    """
    Print the locked experimental protocol represented by the tables.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOCKED EXPERIMENTAL PROTOCOL"
    )

    print(
        "=" * 80
    )

    print(
        f"Qubits: {NUM_QUBITS}"
    )

    print(
        f"Shots: {SHOTS}"
    )

    print(
        f"Seeds: {NUM_SEEDS}"
    )

    print(
        f"Architecture: {ARCHITECTURE}"
    )

    print(
        f"Noise model: {NOISE_MODEL}"
    )

    print(
        "Noise levels: "
        + ", ".join(
            f"{noise:.2f}"
            for noise in NOISE_LEVELS
        )
    )


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_outputs() -> None:
    """
    Confirm that all six manuscript table artifacts were created.
    """

    expected_outputs = [
        TABLE_1_CSV,
        TABLE_1_TEX,
        TABLE_2_CSV,
        TABLE_2_TEX,
        TABLE_3_CSV,
        TABLE_3_TEX,
    ]

    for path in expected_outputs:
        if not path.exists():
            raise RuntimeError(
                "Expected table output was not created: "
                f"{path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                "Table output is empty: "
                f"{path}"
            )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline() -> None:
    print(
        "=" * 80
    )

    print(
        "PUBLICATION TABLE GENERATION"
    )

    print(
        "=" * 80
    )

    print(
        "\nValidating frozen inputs..."
    )

    validate_inputs()

    print(
        "Loading frozen statistical outputs..."
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

    enriched = pd.read_csv(
        ENRICHED_PATH
    )

    # -------------------------------------------------------------------------
    # Validate the frozen main-study structure.
    # -------------------------------------------------------------------------

    expected_rows = (
        NUM_SEEDS
        * len(NOISE_LEVELS)
        * 17
    )

    if len(enriched) != expected_rows:
        raise RuntimeError(
            "Frozen main-study row count does not match "
            "the locked protocol. "
            f"Expected {expected_rows}, found {len(enriched)}."
        )

    observed_seeds = (
        enriched["seed"]
        .nunique()
    )

    if observed_seeds != NUM_SEEDS:
        raise RuntimeError(
            "Unexpected number of seeds in frozen results. "
            f"Expected {NUM_SEEDS}, found {observed_seeds}."
        )

    observed_noise = sorted(
        float(value)
        for value in enriched[
            "noise_level"
        ].unique()
    )

    if not np.allclose(
        observed_noise,
        NOISE_LEVELS,
    ):
        raise RuntimeError(
            "Frozen noise levels do not match "
            "the locked protocol. "
            f"Observed: {observed_noise}"
        )

    print(
        "Frozen experiment verified: "
        f"{len(enriched)} rows, "
        f"{observed_seeds} seeds, "
        f"{len(observed_noise)} noise levels."
    )

    # -------------------------------------------------------------------------
    # Build manuscript tables.
    # -------------------------------------------------------------------------

    print(
        "Building Table I..."
    )

    table_1 = build_table_1()

    print(
        "Building Table II..."
    )

    table_2 = build_table_2(
        descriptive
    )

    print(
        "Building Table III..."
    )

    table_3 = build_table_3(
        paired=paired,
        decoder=decoder,
        measurement_cost=measurement_cost,
    )

    # -------------------------------------------------------------------------
    # Structural checks.
    # -------------------------------------------------------------------------

    if len(table_1) != 9:
        raise RuntimeError(
            "Table I should contain exactly "
            f"9 representations; found {len(table_1)}."
        )

    if len(table_2) != len(
        TABLE_2_REPRESENTATIONS
    ):
        raise RuntimeError(
            "Unexpected number of rows in Table II."
        )

    # Table III:
    #   3 XYZ-vs-parity comparisons
    #   9 XYZ-vs-one-basis comparisons
    #   12 RF-vs-LR comparisons
    #   = 24 rows
    if len(table_3) != 24:
        raise RuntimeError(
            "Table III should contain exactly "
            f"24 selected comparisons; found {len(table_3)}."
        )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    print(
        "Saving CSV and LaTeX tables..."
    )

    save_tables(
        table_1=table_1,
        table_2=table_2,
        table_3=table_3,
    )

    validate_outputs()

    # -------------------------------------------------------------------------
    # Console preview.
    # -------------------------------------------------------------------------

    print_protocol()

    print_tables(
        table_1=table_1,
        table_2=table_2,
        table_3=table_3,
    )

    # -------------------------------------------------------------------------
    # Completion report.
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PUBLICATION TABLES COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSaved:\n{TABLE_1_CSV}"
    )

    print(
        f"\nSaved:\n{TABLE_1_TEX}"
    )

    print(
        f"\nSaved:\n{TABLE_2_CSV}"
    )

    print(
        f"\nSaved:\n{TABLE_2_TEX}"
    )

    print(
        f"\nSaved:\n{TABLE_3_CSV}"
    )

    print(
        f"\nSaved:\n{TABLE_3_TEX}"
    )

    print(
        "\nNo quantum simulation was executed."
    )

    print(
        "No models were retrained."
    )

    print(
        "No new statistical tests were performed."
    )

    print(
        "Tables were generated exclusively from "
        "the frozen main-study outputs."
    )


if __name__ == "__main__":
    run_pipeline()
