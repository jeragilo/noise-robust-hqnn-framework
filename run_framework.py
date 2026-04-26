#!/usr/bin/env python
"""
Entry point for running the full HQNN framework.
"""

import subprocess


def run(cmd):
    print(f"\n>>> Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    run("python -m pipelines.main_hybrid_vs_classical")
    run("python -m pipelines.main_noise_robustness")
    run("python -m pipelines.main_cross_framework_validation")
    run("python -m pipelines.main_full_benchmark_summary")
    run("python -m pipelines.main_framework_capabilities_report")

    print("\nFramework execution complete.")
