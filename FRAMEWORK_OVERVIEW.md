# Noise-Robust Hybrid Quantum Neural Network Framework

## Overview

This repository uses Qiskit, Cirq, PennyLane, and scikit-learn as underlying computational libraries.

The contribution of this project is not to replace these libraries, but to provide a **reusable benchmarking framework** for evaluating hybrid quantum machine learning models under NISQ-era constraints.

---

## What the Framework Adds (Beyond Default Packages)

While existing libraries provide individual algorithms and primitives, this framework provides a **unified evaluation layer** that is not available as a standard workflow in Qiskit Machine Learning or similar packages.

### The framework includes:

1. **Standardized Dataset Handling**
   - Synthetic datasets for controlled experiments
   - Named benchmark datasets:
     - Iris
     - Wisconsin Diagnostic Breast Cancer (WDBC)
   - Consistent preprocessing for quantum-compatible inputs

2. **Hybrid vs Classical Benchmarking**
   - Direct comparison between:
     - Hybrid Quantum Neural Networks (HQNN)
     - Quantum Support Vector Machines (QSVM)
     - Classical models (Logistic Regression, SVM, MLP)
   - Same dataset, same split, same evaluation metrics

3. **Noise Analysis Toolbox**
   - Reusable noise models:
     - Depolarizing
     - Bit-flip
     - Phase-flip
     - Amplitude damping
   - Noise sweeps across multiple probabilities
   - Standardized robustness evaluation

4. **Robustness Metrics (Non-standard Features)**
   - Accuracy drop under noise
   - Relative robustness score
   - Noise degradation slope
   - Training instability (standard deviation)
   - Cross-framework deviation

   These metrics are not provided as a unified system in default quantum libraries.

5. **Cross-Framework Validation**
   - Same experiment evaluated across:
     - Qiskit
     - Cirq
     - PennyLane
   - Measurement of numerical consistency

6. **Pipeline-Based Experimentation**
   - Main pipeline scripts:
     - `main_hybrid_vs_classical.py`
     - `main_noise_robustness.py`
     - `main_cross_framework_validation.py`
   - These pipelines highlight the main contributions of the thesis

7. **Standardized Outputs**
   - JSON files
   - CSV summary tables
   - Accuracy plots
   - Noise curves
   - Heatmaps

   These outputs are directly used in the thesis for analysis.

---

## Structure of the Repository

```text
framework/
  datasets.py
  noise_channels.py
  robustness_metrics.py
  reporting.py
  benchmark_runner.py

pipelines/
  main_hybrid_vs_classical.py
  main_noise_robustness.py
  main_cross_framework_validation.py

demos/
  (individual experiments supporting the framework)

results/framework/
  (aggregated outputs used in the thesis)
