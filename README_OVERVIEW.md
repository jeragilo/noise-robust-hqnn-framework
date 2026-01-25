# Thesis Overview — Noise-Robust Hybrid Quantum Neural Networks (HQNNs)

This repository contains the full experimental codebase for my Master’s thesis,
focused on evaluating the **reliability, robustness, and practical limitations**
of hybrid quantum–classical neural networks (HQNNs) in the NISQ era.

Rather than claiming quantum advantage, this work emphasizes **controlled
experimentation, cross-framework validation, and negative-result analysis** to
understand when and why hybrid quantum ML models fail in practice.

---

## What This Project Includes

- **13 experimental demonstrations** implemented across:
  - Qiskit
  - Cirq
  - PennyLane
- Algorithms and methods:
  - Hybrid Quantum Neural Networks (HQNNs)
  - SPSA-based hybrid training loops
  - Variational Quantum Eigensolver (VQE)
  - Quantum Approximate Optimization Algorithm (QAOA)
  - Quantum Kernel Methods (QSVM)
- System-level studies:
  - Cross-framework correctness validation
  - Noise robustness benchmarking
  - Endianness and measurement consistency analysis
- Applied case studies:
  - Medical risk classification
  - Energy grid optimization
  - Cybersecurity anomaly detection

---

## Key Findings (High-Level)

- Hybrid quantum ML models are **highly sensitive to noise** and optimizer
  instability.
- Classical baselines consistently **outperform HQNNs** on linearly separable
  and applied datasets under realistic conditions.
- Different quantum frameworks exhibit **non-identical noise behavior**, even
  for equivalent circuits.

These findings motivate **reliability-aware hybrid architectures** and caution
against naive deployment of near-term quantum ML systems.

---

## How to Navigate This Repository

- `README.md` — full thesis-oriented documentation
- `README_OVERVIEW.md` — recruiter / reviewer overview (this file)
- `docs/DEMO_SUMMARY.md` — concise summaries of all 13 demonstrations
- `demos/` — executable experimental code
- `docker/` — environment setup
- `THESIS_NOTES.md` — internal thesis notes

---

## Status

✔ Code complete  
✔ Experiments finalized  
🔧 Documentation and diagrams may be expanded

This repository is intended as a **research-grade system**, not a tutorial.
