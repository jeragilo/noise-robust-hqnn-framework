# Framework Capabilities Report

## Purpose

Provide a reusable benchmarking layer for evaluating hybrid quantum machine learning models under NISQ-era constraints.

## Existing Libraries Used

- Qiskit
- Qiskit Aer
- Qiskit Machine Learning
- Cirq
- PennyLane
- scikit-learn

## What Default Libraries Provide

- Quantum circuits
- Quantum simulators
- Quantum kernels
- Classical machine learning models
- Basic plotting and numerical tools

## What This Framework Adds

### Standardized dataset layer
- Code: `framework/datasets.py`
- Description: Provides synthetic, Iris, and Wisconsin Diagnostic Breast Cancer dataset loaders with consistent quantum-compatible preprocessing.

### Noise-analysis toolbox
- Code: `framework/noise_channels.py`
- Description: Centralizes depolarizing, bit-flip, phase-flip, and amplitude-damping noise models with reusable sweep support.

### Robustness metrics
- Code: `framework/robustness_metrics.py`
- Description: Adds accuracy drop, relative robustness score, degradation slope, training instability, and cross-framework deviation metrics.

### Standardized reporting
- Code: `framework/reporting.py`
- Description: Standardizes JSON, CSV, accuracy plots, noise curves, heatmaps, and training-curve outputs.

### Benchmark orchestration
- Code: `framework/benchmark_runner.py`
- Description: Aggregates hybrid-vs-classical, noise-robustness, and cross-framework validation outputs into thesis-ready benchmark rows.

### Main contribution pipelines
- Code: `pipelines/`
- Description: Provides main pipeline files for hybrid-vs-classical comparison, noise robustness, cross-framework validation, and final benchmark summary.

### Cybersecurity anomaly benchmark integration
- Code: `demos/industry/demo11_cyber_anomaly_qiskit.py`
- Description: Combines quantum kernel classification, HQNN training, and classical logistic regression in one domain-inspired cybersecurity benchmark.

## Nonstandard Framework Features

- `accuracy_drop`
- `robustness_score`
- `degradation_slope`
- `training_instability`
- `cross_framework_deviation`
- `final_benchmark_summary`

## Main Pipelines

```bash
python -m pipelines.main_hybrid_vs_classical
```

```bash
python -m pipelines.main_noise_robustness
```

```bash
python -m pipelines.main_cross_framework_validation
```

```bash
python -m pipelines.main_full_benchmark_summary
```

```bash
python -m pipelines.main_framework_capabilities_report
```

