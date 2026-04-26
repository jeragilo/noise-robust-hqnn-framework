"""
Main Pipeline: Full Benchmark Summary

Combines framework-level outputs into one final summary file.
This version handles CSV files with different columns.
"""

from pathlib import Path
import csv
import json

FRAMEWORK_DIR = Path("results/framework")


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def save_flexible_csv(rows, path):
    """
    Save rows even when dictionaries have different columns.
    """
    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline():
    sources = {
        "hybrid_vs_classical": FRAMEWORK_DIR / "hybrid_vs_classical_real_summary.csv",
        "noise_robustness": FRAMEWORK_DIR / "noise_robustness_real_summary.csv",
        "cross_framework": FRAMEWORK_DIR / "cross_framework_summary.csv",
    }

    rows = []

    for category, path in sources.items():
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        for row in read_csv(path):
            row["benchmark_category"] = category
            row["source_file"] = str(path)
            rows.append(row)

    save_flexible_csv(rows, FRAMEWORK_DIR / "final_benchmark_summary.csv")

    save_json(
        {
            "description": "Combined framework-level benchmark summary.",
            "source_files": {k: str(v) for k, v in sources.items()},
            "row_count": len(rows),
            "note": "Rows come from benchmark outputs with different metric columns.",
        },
        FRAMEWORK_DIR / "final_benchmark_summary.json",
    )

    print("Full benchmark summary complete.")
    print("Saved final_benchmark_summary.csv and final_benchmark_summary.json")


if __name__ == "__main__":
    run_pipeline()
