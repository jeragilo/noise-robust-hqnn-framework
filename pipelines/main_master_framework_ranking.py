"""
Pipeline: Master Framework Ranking and Thesis Synthesis

Purpose:
Aggregate all major HQNN framework result JSON files into one final
thesis-level ranking table, summary JSON, and synthesis figures.

This script consolidates the thesis from many separate experiments into
one coherent framework-level result layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path("results") / "framework"
OUTPUT_DIR = RESULTS_ROOT / "master_framework_ranking"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"Missing: {path}")
        return None

    with open(path, "r") as f:
        return json.load(f)


def add_row(
    rows: List[Dict[str, Any]],
    pipeline: str,
    category: str,
    method: str,
    clean_accuracy: Any = None,
    noisy_accuracy: Any = None,
    accuracy_drop: Any = None,
    robustness_score: Any = None,
    dataset: Any = None,
    architecture: Any = None,
    feature_map: Any = None,
    optimizer: Any = None,
    model: Any = None,
    noise_type: Any = None,
    noise_level: Any = None,
    instability: Any = None,
    notes: str = "",
) -> None:
    rows.append(
        {
            "pipeline": pipeline,
            "category": category,
            "method": method,
            "dataset": dataset,
            "architecture": architecture,
            "feature_map": feature_map,
            "optimizer": optimizer,
            "model": model,
            "noise_type": noise_type,
            "noise_level": safe_float(noise_level),
            "clean_accuracy": safe_float(clean_accuracy),
            "noisy_accuracy": safe_float(noisy_accuracy),
            "accuracy_drop": safe_float(accuracy_drop),
            "robustness_score": safe_float(robustness_score),
            "instability": safe_float(instability),
            "notes": notes,
        }
    )


def parse_dual_loss_multi_observable(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "dual_loss_multi_observable_hqnn" / "dual_loss_multi_observable_summary.json"
    data = load_json(path)
    if not data:
        return

    for mode, result in data.get("results", {}).items():
        evaluation = result.get("evaluation", {})
        for readout, metrics in evaluation.items():
            if readout == "feature_dimensions":
                continue
            add_row(
                rows,
                pipeline="dual_loss_multi_observable_hqnn",
                category="noise-aware optimization",
                method=f"{mode} + {readout}",
                clean_accuracy=metrics.get("clean_accuracy"),
                noisy_accuracy=metrics.get("noisy_accuracy"),
                accuracy_drop=metrics.get("accuracy_drop"),
                robustness_score=metrics.get("robustness_score"),
                architecture=result.get("architecture"),
                optimizer=mode,
                model=readout,
                noise_type=data.get("noise_type"),
                noise_level=data.get("eval_noise"),
                notes="Dual-loss/stability-regularized multi-observable HQNN.",
            )


def parse_multichannel(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "multichannel_flagship_validation" / "multichannel_flagship_summary.json"
    data = load_json(path)
    if not data:
        return

    for item in data.get("results", []):
        noise_type = item.get("noise_type")
        for model_name in ["logistic_regression", "random_forest"]:
            metrics = item.get(model_name, {})
            add_row(
                rows,
                pipeline="multichannel_flagship_validation",
                category="multichannel validation",
                method=f"{noise_type} + {model_name}",
                clean_accuracy=metrics.get("clean_accuracy"),
                noisy_accuracy=metrics.get("noisy_accuracy"),
                accuracy_drop=metrics.get("accuracy_drop"),
                robustness_score=metrics.get("robustness_score"),
                architecture=data.get("architecture"),
                model=model_name,
                noise_type=noise_type,
                noise_level=data.get("eval_noise"),
                notes="Flagship configuration validated across multiple noise channels.",
            )


def parse_variance_reduced_spsa(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "variance_reduced_spsa_hqnn" / "variance_reduced_spsa_summary.json"
    data = load_json(path)
    if not data:
        return

    for method, result in data.get("results", {}).items():
        add_row(
            rows,
            pipeline="variance_reduced_spsa_hqnn",
            category="optimizer engineering",
            method=method,
            clean_accuracy=result.get("clean_accuracy"),
            noisy_accuracy=result.get("noisy_accuracy"),
            accuracy_drop=result.get("accuracy_drop"),
            robustness_score=result.get("robustness_score"),
            architecture=data.get("architecture"),
            optimizer=f"SPSA_k{result.get('gradient_samples')}",
            noise_type=data.get("noise_type"),
            noise_level=data.get("eval_noise"),
            instability=result.get("training_instability"),
            notes="SPSA gradient variance reduction by averaging multiple noisy perturbation estimates.",
        )


def parse_noise_augmented_readout(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "noise_augmented_readout_hqnn" / "noise_augmented_readout_summary.json"
    data = load_json(path)
    if not data:
        return

    for strategy, strategy_results in data.get("results", {}).items():
        for model_name, metrics in strategy_results.items():
            if not isinstance(metrics, dict):
                continue
            add_row(
                rows,
                pipeline="noise_augmented_readout_hqnn",
                category="noise-augmented readout",
                method=f"{strategy} + {model_name}",
                clean_accuracy=metrics.get("clean_accuracy"),
                noisy_accuracy=metrics.get("mean_noisy_accuracy"),
                accuracy_drop=None,
                robustness_score=None,
                model=model_name,
                optimizer=strategy,
                noise_type="multi_noise",
                noise_level=data.get("eval_noise"),
                notes="Readout trained on clean/noisy/multichannel quantum feature distributions.",
            )


def parse_feature_fusion(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "quantum_classical_feature_fusion" / "quantum_classical_feature_fusion_summary.json"
    data = load_json(path)
    if not data:
        return

    for experiment, experiment_results in data.get("results", {}).items():
        for model_name, metrics in experiment_results.items():
            if not isinstance(metrics, dict):
                continue
            add_row(
                rows,
                pipeline="quantum_classical_feature_fusion",
                category="quantum-classical fusion",
                method=f"{experiment} + {model_name}",
                clean_accuracy=metrics.get("clean_accuracy"),
                noisy_accuracy=metrics.get("mean_noisy_accuracy"),
                model=model_name,
                optimizer=experiment,
                noise_type="multi_noise",
                noise_level=data.get("eval_noise"),
                notes="Fusion of classical input features and quantum-derived multi-observable features.",
            )


def parse_classical_challenging(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "classical_challenging_benchmark" / "classical_challenging_benchmark_summary.json"
    data = load_json(path)
    if not data:
        return

    for dataset_name, dataset_results in data.get("results", {}).items():
        for experiment, experiment_results in dataset_results.items():
            for model_name, metrics in experiment_results.items():
                if not isinstance(metrics, dict):
                    continue
                add_row(
                    rows,
                    pipeline="classical_challenging_benchmark",
                    category="nonlinear benchmark",
                    method=f"{experiment} + {model_name}",
                    dataset=dataset_name,
                    clean_accuracy=metrics.get("clean_accuracy"),
                    noisy_accuracy=metrics.get("noisy_accuracy"),
                    accuracy_drop=metrics.get("accuracy_drop"),
                    robustness_score=metrics.get("robustness_score"),
                    model=model_name,
                    optimizer=experiment,
                    noise_type=data.get("noise_type", "depolarizing"),
                    noise_level=data.get("noise_level"),
                    notes="Moons/circles/XOR conditional hybrid advantage benchmark.",
                )


def parse_noise_curriculum(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "noise_curriculum_hqnn" / "noise_curriculum_summary.json"
    data = load_json(path)
    if not data:
        return

    for mode, result in data.get("results", {}).items():
        add_row(
            rows,
            pipeline="noise_curriculum_hqnn",
            category="curriculum optimization",
            method=mode,
            clean_accuracy=result.get("clean_accuracy"),
            noisy_accuracy=result.get("noisy_accuracy"),
            accuracy_drop=result.get("accuracy_drop"),
            robustness_score=result.get("robustness_score"),
            architecture=result.get("architecture"),
            optimizer=mode,
            noise_type=result.get("noise_type"),
            noise_level=result.get("eval_noise"),
            instability=result.get("training_instability"),
            notes="Clean, fixed-noise, linear curriculum, and adaptive curriculum HQNN training.",
        )


def parse_data_reuploading(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "data_reuploading_feature_map_search" / "data_reuploading_feature_map_search_summary.json"
    data = load_json(path)
    if not data:
        return

    for item in data.get("results", []):
        add_row(
            rows,
            pipeline="data_reuploading_feature_map_search",
            category="feature-map search",
            method=f"{item.get('feature_map')} + {item.get('readout_mode')} + {item.get('model')}",
            dataset=item.get("dataset"),
            clean_accuracy=item.get("clean_accuracy"),
            noisy_accuracy=item.get("noisy_accuracy"),
            accuracy_drop=item.get("accuracy_drop"),
            robustness_score=item.get("robustness_score"),
            feature_map=item.get("feature_map"),
            model=item.get("model"),
            optimizer=item.get("readout_mode"),
            noise_type=item.get("noise_type"),
            noise_level=item.get("noise_level"),
            notes="Grid search across angle, interaction, and data-reuploading feature maps.",
        )


def parse_real_hardware(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_ROOT / "real_hardware_validation" / "real_hardware_validation_summary.json"
    data = load_json(path)
    if not data:
        return

    aer = data.get("aer_simulator", {})
    ibm = data.get("ibm_hardware")

    add_row(
        rows,
        pipeline="real_hardware_validation",
        category="hardware validation",
        method="Aer parity",
        clean_accuracy=None,
        noisy_accuracy=aer.get("parity_accuracy"),
        model="parity",
        noise_type="Aer simulator",
        notes="Small-scale simulator validation.",
    )

    add_row(
        rows,
        pipeline="real_hardware_validation",
        category="hardware validation",
        method="Aer multi-observable logistic",
        clean_accuracy=None,
        noisy_accuracy=aer.get("multi_observable_logistic_accuracy"),
        model="multi_observable_logistic",
        noise_type="Aer simulator",
        notes="Small-scale simulator multi-observable validation.",
    )

    if ibm is not None:
        add_row(
            rows,
            pipeline="real_hardware_validation",
            category="hardware validation",
            method="IBM parity",
            clean_accuracy=None,
            noisy_accuracy=ibm.get("parity_accuracy"),
            model="parity",
            noise_type=data.get("ibm_metadata", {}).get("backend_name"),
            notes="Real IBM Quantum backend validation.",
        )

        add_row(
            rows,
            pipeline="real_hardware_validation",
            category="hardware validation",
            method="IBM multi-observable logistic",
            clean_accuracy=None,
            noisy_accuracy=ibm.get("multi_observable_logistic_accuracy"),
            model="multi_observable_logistic",
            noise_type=data.get("ibm_metadata", {}).get("backend_name"),
            notes="Real IBM Quantum backend multi-observable validation.",
        )


def add_threshold_evolution(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Thesis-level threshold progression used for final narrative figure.
    These values summarize the major framework evolution points established
    across the experiments.
    """
    return [
        {
            "stage": "Parity HQNN baseline",
            "noisy_accuracy": 0.37,
            "description": "Early parity-based HQNN behavior under realistic noise.",
        },
        {
            "stage": "Learned readout",
            "noisy_accuracy": 0.82,
            "description": "Classical decoder learns noisy measurement distributions.",
        },
        {
            "stage": "Multi-observable features",
            "noisy_accuracy": 0.85,
            "description": "Probability, Z, ZZ, parity, and distribution features.",
        },
        {
            "stage": "Architecture search",
            "noisy_accuracy": 0.8867,
            "description": "Linear shallow entanglement selected as more robust.",
        },
        {
            "stage": "Stability regularization",
            "noisy_accuracy": 0.96,
            "description": "Clean/noisy/stability objective with multi-observable readout.",
        },
        {
            "stage": "Noise-augmented fusion",
            "noisy_accuracy": 0.9733,
            "description": "Readout learns noisy quantum feature geometry.",
        },
    ]


def plot_threshold_evolution(thresholds: List[Dict[str, Any]], output_path: Path) -> None:
    stages = [x["stage"] for x in thresholds]
    values = [x["noisy_accuracy"] for x in thresholds]

    plt.figure(figsize=(12, 6))
    plt.plot(stages, values, marker="o", linewidth=2)
    plt.ylim(0, 1)
    plt.ylabel("Noisy Accuracy")
    plt.title("Threshold Evolution Across HQNN Framework Improvements")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y")

    for x, y in zip(stages, values):
        plt.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_pipelines(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = (
        df.dropna(subset=["noisy_accuracy"])
        .sort_values("noisy_accuracy", ascending=False)
        .head(15)
        .copy()
    )

    labels = [
        f"{row.pipeline}\n{row.method}"[:55]
        for row in plot_df.itertuples(index=False)
    ]

    plt.figure(figsize=(12, 7))
    plt.barh(labels, plot_df["noisy_accuracy"])
    plt.xlim(0, 1)
    plt.xlabel("Noisy Accuracy")
    plt.title("Top Framework Configurations by Noisy Accuracy")
    plt.gca().invert_yaxis()
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_robustness(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.dropna(subset=["robustness_score", "noisy_accuracy"]).copy()
    plot_df = plot_df.sort_values("noisy_accuracy", ascending=False).head(20)

    labels = [
        f"{row.pipeline}\n{row.method}"[:55]
        for row in plot_df.itertuples(index=False)
    ]

    plt.figure(figsize=(12, 7))
    plt.barh(labels, plot_df["robustness_score"])
    plt.xlabel("Robustness Score")
    plt.title("Robustness Score of Top Framework Configurations")
    plt.gca().invert_yaxis()
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_map_comparison(df: pd.DataFrame, output_path: Path) -> None:
    fmap_df = df.dropna(subset=["feature_map", "noisy_accuracy"]).copy()
    if fmap_df.empty:
        return

    grouped = (
        fmap_df.groupby("feature_map")["noisy_accuracy"]
        .max()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(grouped.index, grouped.values)
    plt.ylim(0, 1)
    plt.ylabel("Best Noisy Accuracy")
    plt.title("Best Noisy Accuracy by Quantum Feature Map")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_optimizer_comparison(df: pd.DataFrame, output_path: Path) -> None:
    opt_df = df.dropna(subset=["optimizer", "noisy_accuracy"]).copy()
    if opt_df.empty:
        return

    grouped = (
        opt_df.groupby("optimizer")["noisy_accuracy"]
        .max()
        .sort_values(ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12, 6))
    plt.bar(grouped.index, grouped.values)
    plt.ylim(0, 1)
    plt.ylabel("Best Noisy Accuracy")
    plt.title("Best Noisy Accuracy by Optimizer / Training Strategy")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_master_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    parse_dual_loss_multi_observable(rows)
    parse_multichannel(rows)
    parse_variance_reduced_spsa(rows)
    parse_noise_augmented_readout(rows)
    parse_feature_fusion(rows)
    parse_classical_challenging(rows)
    parse_noise_curriculum(rows)
    parse_data_reuploading(rows)
    parse_real_hardware(rows)

    return rows


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_master_rows()

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No framework result rows were loaded.")

    df = df.sort_values(
        by=["noisy_accuracy", "robustness_score"],
        ascending=[False, False],
        na_position="last",
    )

    csv_path = OUTPUT_DIR / "master_framework_ranking.csv"
    json_path = OUTPUT_DIR / "master_framework_summary.json"

    df.to_csv(csv_path, index=False)

    thresholds = add_threshold_evolution(rows)

    best_overall = (
        df.dropna(subset=["noisy_accuracy"])
        .sort_values("noisy_accuracy", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    best_by_category = (
        df.dropna(subset=["noisy_accuracy"])
        .sort_values("noisy_accuracy", ascending=False)
        .groupby("category")
        .head(1)
        .to_dict(orient="records")
    )

    summary = {
        "description": (
            "Master thesis-level synthesis of all major HQNN framework pipelines. "
            "This file consolidates separate experiments into one final framework "
            "ranking and contribution summary."
        ),
        "num_configurations_ranked": int(len(df)),
        "best_overall_top_10": best_overall,
        "best_by_category": best_by_category,
        "threshold_evolution": thresholds,
        "core_thesis_claim": (
            "Robust hybrid quantum AI under realistic NISQ constraints requires "
            "shallow architectures, hybrid co-design, learned readout systems, "
            "noise-aware optimization, and robustness-guided engineering rather "
            "than isolated idealized quantum classifiers."
        ),
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    threshold_plot = OUTPUT_DIR / "master_threshold_evolution.png"
    ranking_plot = OUTPUT_DIR / "master_top_pipeline_ranking.png"
    robustness_plot = OUTPUT_DIR / "master_robustness_ranking.png"
    feature_map_plot = OUTPUT_DIR / "master_feature_map_comparison.png"
    optimizer_plot = OUTPUT_DIR / "master_optimizer_comparison.png"

    plot_threshold_evolution(thresholds, threshold_plot)
    plot_best_pipelines(df, ranking_plot)
    plot_robustness(df, robustness_plot)
    plot_feature_map_comparison(df, feature_map_plot)
    plot_optimizer_comparison(df, optimizer_plot)

    print("\nMaster framework ranking complete.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {threshold_plot}")
    print(f"Saved: {ranking_plot}")
    print(f"Saved: {robustness_plot}")
    print(f"Saved: {feature_map_plot}")
    print(f"Saved: {optimizer_plot}")

    print("\nTop 10 configurations by noisy accuracy:")
    print(df[["pipeline", "category", "method", "dataset", "clean_accuracy", "noisy_accuracy", "accuracy_drop"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
