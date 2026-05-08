# Noise-Robust Hybrid Quantum Neural Networks:
## Thesis Results Summary

## Core Thesis Claim

This thesis demonstrates that hybrid quantum neural network (HQNN) robustness in NISQ environments depends strongly on:

- readout strategy,
- quantum feature extraction,
- entanglement architecture,
- noise-aware optimization objectives,
- and robustness-oriented training methods.

Rather than treating noise only as a post-training evaluation condition, this framework embeds noise awareness directly into HQNN optimization, architecture selection, and hybrid quantum-classical inference.

The main contribution is a reusable HQNN robustness optimization framework.

---

# Core Framework Contributions

## 1. Learned Readout HQNN

Baseline HQNN systems using fixed parity readout performed poorly under noisy evaluation.

The framework replaced fixed parity readout with learned classical decoding of quantum measurement distributions.

### Result

| Method | Noisy Accuracy |
|---|---:|
| Fixed parity HQNN | ~0.37 |
| Learned-readout HQNN | ~0.82 |

### Main Insight

The bottleneck was not necessarily the quantum representation itself, but the simplistic readout mechanism.

---

## 2. Multi-Observable Quantum Feature Extraction

The framework expanded quantum feature extraction beyond parity and raw probabilities.

Extracted features included:

- full probability distributions,
- single-qubit Z expectations,
- pairwise ZZ correlations,
- parity expectation,
- entropy/statistical descriptors.

### Result

| Method | Noisy Accuracy |
|---|---:|
| Multi-observable Logistic Regression | ~0.82 |
| Multi-observable Random Forest | ~0.85 |

### Main Insight

Richer quantum observable extraction substantially improves HQNN robustness and classification performance.

---

## 3. Architecture-Aware HQNN Optimization

The framework tested multiple HQNN entanglement layouts:

- none
- linear
- ring
- full

### Best Result

| Architecture | Readout | Noisy Accuracy |
|---|---|---:|
| Linear | Random Forest | ~0.89 |

### Main Insight

Entanglement architecture significantly affects HQNN robustness under noise.

---

## 4. Noise Sweep Validation

The optimized HQNN configuration was tested across increasing depolarizing noise levels.

### Result

| Noise Level | Accuracy |
|---|---:|
| 0.00 | ~0.85 |
| 0.05 | ~0.85 |
| 0.10 | ~0.81 |

### Main Insight

The optimized HQNN configuration maintained strong performance under increasing NISQ-style noise.

---

## 5. Repeated-Trial Validation

The framework validated reproducibility across randomized trials.

### Aggregate Results

| Metric | Value |
|---|---:|
| Mean noisy accuracy | ~0.81 |
| Maximum noisy accuracy | ~0.94 |
| Mean robustness score | ~0.92 |

### Main Insight

The optimized HQNN improvement is not merely a single-run artifact.

---

## 6. Statistical Validation Framework

The framework added statistical evaluation capabilities:

- confidence intervals,
- stability ranking,
- robustness ranking,
- coefficient of variation,
- consistency metrics.

### Main Insight

The thesis evolved from isolated demos into a reusable HQNN robustness evaluation framework.

---

## 7. Dual-Loss Noise-Aware Optimization

The framework introduced true noise-aware optimization objectives.

Training objectives combined:

- clean loss,
- noisy loss,
- stability regularization.

### Main Insight

Noise awareness can be embedded directly into HQNN optimization rather than treated only as evaluation.

---

## 8. Stability-Regularized HQNN Training

The framework penalized disagreement between clean and noisy predictions.

### Best Result

| Method | Clean Accuracy | Noisy Accuracy |
|---|---:|---:|
| Stability-Regularized Multi-Observable RF | ~0.96 | ~0.96 |

### Main Insight

Stability-oriented optimization can substantially improve HQNN robustness under noisy evaluation.

---

## 9. Multi-Channel Noise Validation

The flagship HQNN configuration was validated across:

- depolarizing noise,
- bit-flip noise,
- phase-flip noise,
- amplitude damping.

### Results

| Noise Channel | RF Noisy Accuracy |
|---|---:|
| Depolarizing | ~0.95 |
| Bit flip | ~0.95 |
| Phase flip | ~0.95 |
| Amplitude damping | ~0.93 |

### Main Insight

The optimized HQNN method generalizes across multiple NISQ-relevant noise channels.

---

## 10. Variance-Reduced Noise-Aware SPSA

The framework introduced optimizer-level variance reduction.

Instead of using one noisy SPSA gradient estimate per update, the method averaged multiple noisy gradient realizations.

### Result

| Method | Noisy Accuracy |
|---|---:|
| Standard SPSA | ~0.40 |
| Variance-Reduced SPSA (k=3) | ~0.59 |

### Main Insight

Moderate gradient averaging improved noisy HQNN optimization stability.

---

# Overall Thesis Contribution

The thesis evolved from simple HQNN benchmarking into:

- a reusable robustness optimization framework,
- a noise-aware HQNN training methodology,
- a hybrid quantum-classical feature extraction framework,
- a robustness-oriented architecture selection framework,
- and a generalized NISQ evaluation methodology.

The strongest conceptual finding is:

> HQNN performance under NISQ noise depends strongly on how noisy quantum information is extracted, interpreted, optimized, and stabilized.

---

# Final Best Result

| Configuration | Clean Accuracy | Noisy Accuracy |
|---|---:|---:|
| Stability-Regularized Multi-Observable RF HQNN | ~0.96 | ~0.96 |

---

# Repository

https://github.com/jeragilo/noise-robust-hqnn-framework
