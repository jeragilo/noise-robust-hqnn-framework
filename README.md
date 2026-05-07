# Noise-Robust Hybrid Quantum Neural Networks Framework

This repository contains the experimental codebase for my Master’s thesis:

**Noise-Robust Hybrid Quantum Neural Networks: A Framework for Scalable Quantum AI in the NISQ Era**

The project investigates the reliability, robustness, and practical limitations of hybrid quantum–classical neural networks under simulated NISQ-era noise.

This repository is structured as a reusable benchmarking and optimization framework, not only as a collection of independent demos.

---

## Flagship Result: Stability-Regularized Multi-Observable HQNN

The strongest configuration combines:

- Stability-regularized noise-aware training
- Multi-observable quantum feature extraction
- Linear HQNN architecture
- Random Forest learned classical readout

Result under depolarizing noise:

| Metric | Value |
|---|---:|
| Clean accuracy | 0.9600 |
| Noisy accuracy | 0.9600 |
| Accuracy drop | 0.0000 |
| Robustness score | 1.0000 |
| Gain over random parity baseline | +0.5067 |

This supports the main thesis claim that HQNN robustness improves when noise awareness is embedded into the optimization process and richer quantum measurement information is preserved through learned hybrid readouts.

---

## Main Thesis Claim

Naive HQNN models with fixed parity readout can perform poorly because they compress quantum measurement information into a single decision signal. This framework shows that HQNN performance can be substantially improved by combining:

1. Noise-aware training objectives  
2. Stability regularization  
3. Multi-observable quantum feature extraction  
4. Architecture search  
5. Learned classical readouts  
6. Repeated-trial and statistical validation  

The framework moves the thesis from simple HQNN benchmarking toward a reusable method for discovering noise-robust hybrid quantum-classical learning configurations.

---

## Key Algorithmic Contributions

### 1. Learned Classical Readout HQNN

Instead of using a fixed parity threshold, the circuit measurement distribution is used as a quantum feature representation for a learned classical readout.

Best observed learned-readout result:

| Method | Clean Accuracy | Noisy Accuracy |
|---|---:|---:|
| Fixed parity readout | 0.3444 | 0.3667 |
| Learned logistic readout | 0.8333 | 0.8222 |

---

### 2. Multi-Observable HQNN Readout

The framework extracts richer quantum-derived features:

- Full bitstring probabilities
- Single-qubit Z expectations
- Pairwise ZZ correlations
- Global parity expectation
- Probability-distribution statistics

This produced a 31-dimensional quantum-derived feature vector.

Best observed multi-observable result:

| Method | Clean Accuracy | Noisy Accuracy |
|---|---:|---:|
| Multi-observable Logistic Regression | 0.8000 | 0.8200 |
| Multi-observable Random Forest | 0.8733 | 0.8533 |

---

### 3. Architecture Search for HQNN Robustness

The framework compares different entanglement structures:

- No entanglement
- Linear entanglement
- Ring entanglement
- Full entanglement

Best architecture-search result:

| Architecture | Readout | Clean Accuracy | Noisy Accuracy |
|---|---|---:|---:|
| Linear | Random Forest | 0.8933 | 0.8867 |

---

### 4. Best-Architecture Noise Sweep

The optimized linear HQNN configuration was tested across increasing depolarizing noise levels.

| Noise Level | Accuracy |
|---:|---:|
| 0.00 | 0.8467 |
| 0.01 | 0.8467 |
| 0.03 | 0.8467 |
| 0.05 | 0.8467 |
| 0.07 | 0.8200 |
| 0.10 | 0.8133 |

This shows that the optimized configuration remains stable across moderate noise levels.

---

### 5. Dual-Loss and Stability-Regularized HQNN Training

The framework implements noise-aware objective functions that combine clean behavior, noisy behavior, and stability under perturbation.

Training objectives include:

- Standard clean-loss training
- Noise-aware training
- Dual-loss training
- Stability-regularized training

Best observed result:

| Training Mode | Readout | Clean Accuracy | Noisy Accuracy | Accuracy Drop |
|---|---|---:|---:|---:|
| Stability-regularized | Multi-observable Random Forest | 0.9600 | 0.9600 | 0.0000 |

---

## Framework Overview

The framework uses:

- Qiskit
- Qiskit Aer
- Qiskit Machine Learning
- Cirq
- PennyLane
- scikit-learn
- NumPy
- Matplotlib

The contribution of this project is the reusable evaluation, optimization, reporting, and robustness-analysis layer built around those libraries.

---

## Framework Features

### Standardized Dataset Handling

- Synthetic classification data
- Iris
- Wisconsin Diagnostic Breast Cancer dataset
- Quantum-compatible low-dimensional preprocessing

### Noise-Analysis Toolbox

Supported noise models:

- Depolarizing noise
- Bit-flip noise
- Phase-flip noise
- Amplitude damping

### Robustness Metrics

The framework includes reusable metrics:

- `accuracy_drop`
- `robustness_score`
- `degradation_slope`
- `training_instability`
- `cross_framework_deviation`

### Benchmark Pipelines

The framework includes pipelines for:

- Hybrid vs classical comparison
- Noise robustness
- Cross-framework validation
- Learned-readout HQNN evaluation
- Multi-observable HQNN evaluation
- Architecture search
- Best-architecture noise sweep
- Repeated-trial validation
- Statistical validation
- Dual-loss noise-aware training
- Stability-regularized multi-observable HQNN

### Standardized Outputs

The framework generates:

- JSON summaries
- CSV summaries
- Accuracy plots
- Noise curves
- Heatmaps
- Statistical validation reports

---

## Repository Structure

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
  main_full_benchmark_summary.py
  main_framework_capabilities_report.py
  main_training_mode_comparison.py
  main_learned_readout_hqnn.py
  main_multi_observable_hqnn.py
  main_architecture_search_hqnn.py
  main_best_architecture_noise_sweep.py
  main_best_architecture_repeated_trials.py
  main_statistical_validation.py
  main_dual_loss_noise_aware_hqnn.py
  main_dual_loss_multi_observable_hqnn.py

demos/
  core/
  industry/

results/
  framework/

env/
  requirements.txt

run_framework.py
Running the Framework

Run the full framework:

python run_framework.py

Run individual framework pipelines:

PYTHONPATH=. python pipelines/main_hybrid_vs_classical.py
PYTHONPATH=. python pipelines/main_noise_robustness.py
PYTHONPATH=. python pipelines/main_cross_framework_validation.py
PYTHONPATH=. python pipelines/main_full_benchmark_summary.py
PYTHONPATH=. python pipelines/main_framework_capabilities_report.py

Run advanced HQNN optimization pipelines:

PYTHONPATH=. python pipelines/main_learned_readout_hqnn.py
PYTHONPATH=. python pipelines/main_multi_observable_hqnn.py
PYTHONPATH=. python pipelines/main_architecture_search_hqnn.py
PYTHONPATH=. python pipelines/main_best_architecture_noise_sweep.py
PYTHONPATH=. python pipelines/main_best_architecture_repeated_trials.py
PYTHONPATH=. python pipelines/main_statistical_validation.py
PYTHONPATH=. python pipelines/main_dual_loss_noise_aware_hqnn.py
PYTHONPATH=. python pipelines/main_dual_loss_multi_observable_hqnn.py
Demonstration Ecosystem

The project includes a 13-demo experimental ecosystem across Qiskit, Cirq, and PennyLane.

Core Demos
HQNN Toy Classifier
VQE Energy Minimization
QAOA MaxCut
QSVM Anomaly Detection
Noise-Robust HQNN
Cross-Framework Noise Benchmark
Cross-Platform Parity Consistency
HQNN Training Loop with SPSA
Industry-Inspired Demos
Medical Risk Classification
Energy Grid Optimization
Cybersecurity Anomaly Detection
HQNN Explainability
Cross-Noise Robustness Heatmap

Run an individual demo from the repository root:

PYTHONPATH=. python demos/core/demo05_hqnn_noise_robust_qiskit.py

Cybersecurity demo:

PYTHONPATH=. python demos/industry/demo11_cyber_anomaly_qiskit.py
Environment Setup
conda create -n hqnn python=3.11 -y
conda activate hqnn
pip install -r env/requirements.txt
Thesis-Relevant Interpretation

The results suggest that HQNN performance is not determined only by the quantum circuit itself. It depends strongly on the full hybrid pipeline:

how noise is inserted into training,
how measurement information is extracted,
how the classical readout interprets quantum features,
how entanglement architecture is selected,
and how robustness is validated statistically.

The strongest result shows that a stability-regularized, multi-observable, learned-readout HQNN can preserve performance under simulated depolarizing noise, achieving 0.9600 clean accuracy and 0.9600 noisy accuracy in the reported experiment.

Status

The repository currently includes:

Framework layer
Pipeline layer
Core demos
Industry demos
Benchmark outputs
Advanced HQNN optimization pipelines
Statistical validation outputs
Thesis-ready figures and JSON summaries
Contact

GitHub: https://github.com/jeragilo/

