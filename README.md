Noise-Robust Hybrid Quantum Neural Networks (HQNNs)

This repository contains the complete experimental codebase for my Master’s thesis,
which investigates the reliability, robustness, and practical limitations of
hybrid quantum–classical neural networks (HQNNs) in the NISQ era.

The project consists of a 13-demo experimental ecosystem implemented across
Qiskit, Cirq, and PennyLane, combining:
- hybrid quantum learning (HQNNs, SPSA)
- variational algorithms (VQE, QAOA)
- quantum kernel methods (QSVM)
- cross-framework correctness validation
- noise robustness benchmarking
- applied case studies (healthcare, energy, cybersecurity)

📌 This repository is intended as a research-grade system.

Demonstration Ecosystem (13 Demos)

This repository includes thirteen experimental demonstrations used in the thesis. Each demo validates a different part of the hybrid architecture.

Core Demos

HQNN Toy Classifier (Qiskit)

VQE Energy Minimization (PennyLane)

QAOA MaxCut (Cirq)

QSVM Anomaly Detection (Qiskit ML)

Noise-Robust HQNN (Qiskit)

Cross-Framework Noise Benchmark (Qiskit/Cirq/PennyLane)

Endianness Parity Consistency (Qiskit/Cirq/PennyLane)

Hybrid HQNN Training Loop with SPSA

Industry-Inspired Demos

Medical Risk Classification (HQNN vs Classical)

Energy Grid Optimization using QAOA (Cirq)

Cybersecurity Anomaly Detection (QSVM + HQNN)

HQNN Explainability and Sensitivity Analysis

Cross-Noise Robustness Heatmap (Qiskit/Cirq/PennyLane)

Full descriptions of each demonstration are provided in the Demo_Descriptions.pdf document included in the thesis package.

How to Run a Demo

Each demo is contained inside the demos/ folder.

For example:

cd demos/core

python demo01_hqnn_toy_classifier.py


Or for an industry demo:

cd demos/industry

python demo10_energy_qaoa.py


Some demos require additional dependencies. Instructions will be added as diagrams and detailed documentation are integrated.

Environment Setup

A simple environment setup is provided:

python3 -m venv env

source env/bin/activate

pip install -r requirements.txt

Pseudocode

All pseudocode files for the thirteen demos are included in the thesis submission ZIP under:

4_Pseudocode/


They are not duplicated in this repository to avoid clutter and maintain a clean structure.

Documentation

Complete documentation is contained in:

Technical Manuscript (PDF)

Thesis Pre-Draft (~300 pages)

Demo Descriptions (PDF)

Slide Deck (50 slides)

These are included in the thesis submission package sent to the advisor.

Status

The codebase is complete.
The next update will include:

architecture diagrams

circuit diagrams

BPMN-style workflow diagrams

demonstration figures and plots

Contact

For inquiries or questions about the project:


GitHub: https://github.com/jeragilo/
