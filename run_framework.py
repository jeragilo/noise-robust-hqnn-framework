#!/usr/bin/env python
"""
Main entry point for the Noise-Robust HQNN Benchmarking Framework.

This script runs the framework-level pipelines that highlight the main
contributions of the project:

1. Hybrid vs classical benchmarking
2. Noise robustness analysis
3. Cross-framework validation
4. Final benchmark aggregation
5. Framework capabilities reporting
"""

import subprocess


def run(command: str) -> None:
    print(f"\n>>> Running: {command}")
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    run("python -m pipelines.main_hybrid_vs_classical")
    run("python -m pipelines.main_noise_robustness")
    run("python -m pipelines.main_cross_framework_validation")
    run("python -m pipelines.main_full_benchmark_summary")
    run("python -m pipelines.main_framework_capabilities_report")

    print("\nFramework execution complete.")
    print("Framework outputs saved in results/framework/")
