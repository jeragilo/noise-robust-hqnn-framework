# Noise-Robust Hybrid Quantum Neural Networks

**Master’s thesis research framework for reliable hybrid quantum–classical learning under NISQ noise.**

This repository contains the experimental codebase developed for my M.S. Computer Science thesis at East Carolina University:

**Noise-Robust Hybrid Quantum Neural Networks: A Framework for Scalable Quantum AI in the NISQ Era**

The project studies how **noise-aware training, measurement design, entanglement architecture, and learned classical readout** affect the reliability of hybrid quantum neural networks (HQNNs). It is organized as a reusable benchmarking and optimization framework rather than a collection of isolated demonstrations.

## Research Highlights

| Experiment | Clean Accuracy | Noisy Accuracy |
|---|---:|---:|
| Fixed parity readout | 0.3444 | 0.3667 |
| Learned logistic readout | 0.8333 | 0.8222 |
| Multi-observable Random Forest | 0.8733 | 0.8533 |
| Best architecture search: Linear + RF | 0.8933 | 0.8867 |
| Stability-regularized multi-observable RF* | 0.9600 | 0.9600 |

\*Best observed configuration in the reported experiment. Results are experimental and configuration-dependent; they are not claims of general quantum advantage.

The central empirical finding is that HQNN performance depends strongly on the **full hybrid pipeline**, not only the parameterized quantum circuit. Preserving richer measurement information and learning the classical decision rule produced substantially stronger evaluated performance than compressing circuit outputs through a fixed parity rule.

## Multi-Channel Noise Validation

The flagship configuration was evaluated at `eval_noise = 0.05` across four simulated NISQ-relevant channels:

| Noise Channel | Clean Accuracy | Noisy Accuracy | Accuracy Drop | Robustness Score |
|---|---:|---:|---:|---:|
| Depolarizing | 0.9600 | 0.9467 | 0.0133 | 0.9861 |
| Bit flip | 0.9467 | 0.9467 | 0.0000 | 1.0000 |
| Phase flip | 0.9533 | 0.9533 | 0.0000 | 1.0000 |
| Amplitude damping | 0.9600 | 0.9267 | 0.0333 | 0.9653 |

These experiments test whether the selected hybrid design remains stable across more than one simulated noise condition.

## Main Contributions

### Learned classical readout

Instead of reducing quantum measurements immediately to a fixed parity decision, the framework uses the circuit measurement distribution as a quantum-derived feature representation for a learned classical model.

### Multi-observable quantum features

The framework extracts a 31-dimensional feature representation containing:

- full bitstring probabilities
- single-qubit Z expectations
- pairwise ZZ correlations
- global parity expectation
- probability-distribution statistics

### Entanglement architecture search

HQNN configurations are compared across:

- no entanglement
- linear entanglement
- ring entanglement
- full entanglement

The best reported architecture-search configuration used **linear entanglement with a Random Forest readout**, reaching 0.8933 clean and 0.8867 noisy accuracy in the evaluated experiment.

### Noise-aware optimization

Implemented training objectives include:

- standard clean-loss training
- noise-aware training
- dual-loss training
- stability-regularized training

The objective is to search for parameter settings that remain useful under simulated perturbations rather than optimizing exclusively for ideal circuit behavior.

### Repeated-trial and statistical validation

The framework includes repeated experiments and statistical reporting so that conclusions are not based solely on a single stochastic training run.

## Noise Sweep

The optimized linear HQNN configuration was evaluated across increasing depolarizing-noise levels:

| Noise Level | Accuracy |
|---:|---:|
| 0.00 | 0.8467 |
| 0.01 | 0.8467 |
| 0.03 | 0.8467 |
| 0.05 | 0.8467 |
| 0.07 | 0.8200 |
| 0.10 | 0.8133 |

Within this experiment, the selected configuration remained stable through moderate simulated noise before degrading at higher levels.

## Technology Stack

**Quantum:** Qiskit · Qiskit Aer · Qiskit Machine Learning · Cirq · PennyLane  
**Machine Learning:** scikit-learn  
**Scientific Computing:** NumPy · SciPy · Pandas  
**Analysis & Visualization:** Matplotlib

## Framework Capabilities

### Datasets

- synthetic classification data
- Iris
- Wisconsin Diagnostic Breast Cancer dataset
- low-dimensional preprocessing compatible with the evaluated quantum circuits

### Simulated noise models

- depolarizing
- bit flip
- phase flip
- amplitude damping

### Robustness metrics

- `accuracy_drop`
- `robustness_score`
- `degradation_slope`
- `training_instability`
- `cross_framework_deviation`

### Experimental pipelines

- hybrid vs. classical comparison
- noise robustness evaluation
- cross-framework validation
- learned-readout HQNN evaluation
- multi-observable HQNN evaluation
- architecture search
- best-architecture noise sweep
- repeated-trial validation
- statistical validation
- dual-loss noise-aware training
- stability-regularized multi-observable HQNN

### Outputs

- JSON summaries
- CSV summaries
- accuracy plots
- noise curves
- heatmaps
- statistical validation reports

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

tests/
  test_datasets.py
  test_robustness_metrics.py

demos/
  core/
  industry/

results/
  framework/

env/
  requirements.txt
  requirements-dev.txt

docs/
  REPRODUCIBILITY.md

.github/workflows/
  tests.yml

CITATION.cff
run_framework.py
```

## Quick Start

Create the environment:

```bash
conda create -n hqnn python=3.11 -y
conda activate hqnn
pip install -r env/requirements.txt
```

Run the framework from the repository root:

```bash
python run_framework.py
```

Run an individual experiment:

```bash
PYTHONPATH=. python pipelines/main_learned_readout_hqnn.py
PYTHONPATH=. python pipelines/main_multi_observable_hqnn.py
PYTHONPATH=. python pipelines/main_architecture_search_hqnn.py
PYTHONPATH=. python pipelines/main_best_architecture_noise_sweep.py
PYTHONPATH=. python pipelines/main_statistical_validation.py
```

## Testing

The repository includes lightweight deterministic tests for core robustness metrics and dataset preprocessing. These tests are intentionally separated from long-running quantum optimization experiments so that code-level verification remains fast.

Install the development dependencies and run the test suite:

```bash
pip install -r env/requirements-dev.txt
pytest -q tests
```

GitHub Actions is configured to run the same test suite on pushes and pull requests targeting `main`.

## Reproducibility

Detailed reproduction instructions are available in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), including environment setup, testing, Docker usage, individual pipeline execution, and guidance for interpreting stochastic experimental results.

The repository separates research/runtime dependencies from development/test dependencies:

- `env/requirements.txt` — research and runtime environment
- `env/requirements-dev.txt` — runtime environment plus test tooling

## Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff). If this software is used in research, please cite the repository and the associated thesis or publication when available.

## Demonstration Ecosystem

The repository also contains 13 experimental demonstrations across Qiskit, Cirq, and PennyLane, including:

**Core:** HQNN classification, VQE, QAOA, QSVM anomaly detection, noise-robust HQNN evaluation, cross-framework noise benchmarking, cross-platform parity consistency, and SPSA training.

**Application-oriented:** medical-risk classification, energy-grid optimization, cybersecurity anomaly detection, HQNN explainability, and cross-noise robustness analysis.

These demonstrations complement the primary HQNN research pipelines; they are not presented as independent claims of practical quantum advantage.

## Research Interpretation

The experiments support a hybrid-first interpretation of near-term quantum machine learning: circuit design matters, but so do the interfaces around the circuit. In the evaluated configurations, robustness was strongly influenced by how measurement information was retained, how the classical readout used quantum-derived features, how entanglement was structured, and whether noise was represented during optimization.

The repository therefore treats the quantum circuit as one component of a larger learning system and evaluates robustness at the **pipeline level**.

## Reproducibility Notes

Results in this repository are experimental outputs from specific datasets, random seeds, circuit configurations, shot counts, noise settings, and classical readout models. Reported values should be interpreted in that experimental context rather than as universal performance guarantees or evidence of general quantum advantage.

## Author

**Jesús Gil**  
M.S. Computer Science, East Carolina University  
[GitHub](https://github.com/jeragilo) · [LinkedIn](https://www.linkedin.com/in/jesusrgil)
