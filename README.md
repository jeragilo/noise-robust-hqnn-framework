Noise-Robust Hybrid Quantum Neural Networks Framework

This repository contains the experimental codebase for my Master’s thesis, which investigates the reliability, robustness, and practical limitations of hybrid quantum–classical neural networks in the NISQ era.

This project is structured as a reusable benchmarking framework, not only as a collection of independent demos.

Framework Overview: Key Contribution

The framework uses Qiskit, Qiskit Aer, Qiskit Machine Learning, Cirq, PennyLane, and scikit-learn as underlying computational libraries. The contribution of this project is the reusable evaluation layer built around those libraries.

The framework adds:
Standardized dataset handling
Synthetic data
Iris
Wisconsin Diagnostic Breast Cancer (WDBC)
Quantum-compatible preprocessing
Designed for low-dimensional quantum input
Noise-analysis toolbox
Depolarizing
Bit-flip
Phase-flip
Amplitude damping
Robustness metrics (nonstandard features)
accuracy_drop
robustness_score
degradation_slope
training_instability
cross_framework_deviation
Benchmark pipelines
Hybrid vs Classical
Noise robustness
Cross-framework validation
Domain-inspired benchmarks
Cybersecurity anomaly detection
Medical classification
Energy optimization
Standardized outputs
JSON
CSV
Accuracy plots
Noise curves
Heatmaps
Framework Structure

Main framework code:

framework/
  datasets.py
  noise_channels.py
  robustness_metrics.py
  reporting.py
  benchmark_runner.py

Main orchestration pipelines:

pipelines/
  main_hybrid_vs_classical.py
  main_noise_robustness.py
  main_cross_framework_validation.py
  main_full_benchmark_summary.py
  main_framework_capabilities_report.py

Framework outputs:

results/framework/
Running the Framework

Run the full framework:

python run_framework.py

Or run pipelines individually:

python -m pipelines.main_hybrid_vs_classical
python -m pipelines.main_noise_robustness
python -m pipelines.main_cross_framework_validation
python -m pipelines.main_full_benchmark_summary
python -m pipelines.main_framework_capabilities_report
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
HQNN Training Loop (SPSA)
Industry-Inspired Demos
Medical Risk Classification
Energy Grid Optimization
Cybersecurity Anomaly Detection
HQNN Explainability
Cross-Noise Robustness Heatmap
Running Individual Demos

Run from the repo root:

python -m demos.core.demo05_hqnn_noise_robust_qiskit

Cybersecurity demo:

python -m demos.industry.demo11_cyber_anomaly_qiskit
Environment Setup
conda create -n hqnn python=3.11 -y
conda activate hqnn
pip install -r env/requirements.txt
Framework-Specific Features

This framework provides components not available as a unified workflow in default quantum libraries:

accuracy_drop
robustness_score
degradation_slope
training_instability
cross_framework_deviation
framework-level CSV/JSON reporting
cross-framework validation summaries
hybrid-versus-classical benchmark summaries
Documentation

Includes:

Technical Manuscript
Thesis Draft
Demo Descriptions
Slide Deck
Pseudocode Files
Status

The repository includes:

Framework layer
Pipeline layer
Core demos
Industry demos
Benchmark outputs
Contact

GitHub: https://github.com/jeragilo/
