# Beyond Parity — Experimental Methods and Evidence Guide

**Manuscript:** *Beyond Parity: Measurement Representation and Decoder Capacity in Noise-Robust Hybrid Quantum Neural Networks*  
**Authors:** Jesus Gil and David Hart (East Carolina University)  
**Status:** Submitted to *Quantum Machine Intelligence* (Springer Nature); not accepted or published as of this documentation.

> This guide summarizes the study described in the submitted manuscript. It is **not** a claim that the complete Beyond Parity experimental dataset, raw IBM Quantum jobs, or a fully reproducible run package is publicly available in this repository. The broader thesis experiments in this repository should not be conflated with the distinct Beyond Parity study.

## Research question

How do measurement representation and classical decoder capacity affect hybrid quantum neural network (HQNN) classification and noise robustness when the quantum feature generator is held fixed? How much extra benefit justifies additional measurement bases and shots?

## Experimental design

- **Quantum feature generator:** Four qubits; input encoding with RY and RZ rotations, seed-specific randomly initialized parameters, and ring CZ entanglement. Quantum parameters are **held fixed**, not trained end-to-end, to isolate readout effects.
- **Readouts (nine):** Fixed parity (R0), learned parity (R0L), Single-Z (R1), All-Z (R2), XYZ expectations (R3), computational-basis probabilities (R4), Z/ZZ correlations (R5), probabilities plus Z/ZZ (R6), and a 31-feature full representation (R7).
- **Classical decoders:** Logistic regression and random forest, using fixed model settings rather than representation-specific tuning.
- **Primary synthetic benchmark:** 500 examples per seed; stratified 60/20/20 train/validation/test; 20 matched seeds × five simulated depolarizing-noise levels `p = {0, 0.02, 0.05, 0.10, 0.20}` (100 seed–noise conditions).
- **Binary Iris validation:** Setosa versus versicolor; 20 matched seeds and three simulated noise levels (`0, 0.10, 0.20`).
- **Sampling:** One-basis readouts use Z-basis records with 1,024 shots per sample; XYZ uses X, Y, and Z bases with 1,024 shots each (3,072 shots per sample). All Z-only representations are derived from a common Z-basis measurement record for matched comparisons.
- **Statistics:** Paired seed-level comparisons, 95% confidence intervals, paired t-tests, Wilcoxon checks, effect sizes, and Holm correction. Lack of statistical significance is not evidence of equivalence.

## IBM Marrakesh physical-device validation

The manuscript reports an **exploratory five-seed binary Iris experiment** on the IBM Marrakesh quantum processor, not a controlled physical-hardware noise sweep.

| Study characteristic | Reported value |
| --- | --- |
| Classification task | Binary Iris (setosa versus versicolor) |
| Circuit | Four-qubit feature generator, seed-specific frozen parameters |
| Per-seed data | 60 training, 20 validation, 20 held-out test examples |
| Measurement bases | X, Y, Z |
| Shots per basis | 1,024 |
| Circuit executions | **1,500 total** (300 per seed × five seeds) |
| Measurement shots | **1,536,000 total** |

### Reported hardware results

| Readout | Decoder | Mean held-out test accuracy | 95% CI | Measurement bases |
| --- | --- | ---: | --- | ---: |
| Fixed parity | Fixed threshold | 0.530 | [0.351, 0.709] | 1 |
| All-Z | Random forest | 0.960 | [0.879, 1.000] | 1 |
| XYZ | Random forest | 0.970 | [0.936, 1.000] | 3 |

All-Z minus fixed parity: **+0.430** mean accuracy (Holm-adjusted `p = 0.0047`). XYZ minus fixed parity: **+0.440** (Holm-adjusted `p = 0.0047`). XYZ minus All-Z: **+0.010** (95% CI [-0.058, 0.078], Holm-adjusted `p = 0.704`): no statistically significant difference detected in this small exploratory comparison.

**Interpretation:** In this evaluated task, preserving informative measurement features mattered much more than moving from a strong one-basis readout to a three-basis readout. These results do not demonstrate general quantum advantage, statistical equivalence of All-Z and XYZ, or performance on larger or different tasks.

## Simulation results and limits

On the primary synthetic benchmark with random-forest decoding, rich representations achieved approximately **0.83–0.87** mean test accuracy under reported conditions, versus approximately **0.51–0.55** for learned parity. The smaller XYZ advantage over strong one-basis representations was detected under clean/moderate noise but not at the highest tested noise level. Binary Iris reproduced the large parity-versus-rich-readout separation, but not a consistent additional XYZ advantage.

The quantum feature generator was intentionally frozen; the results do **not** evaluate fully end-to-end quantum-parameter training. The principal controlled noise model was simulated depolarizing noise, while the hardware experiment was exploratory and subject to real-device variation.

## Evidence availability and reproduction status

The current repository contains the **broader master's-thesis HQNN framework**, including research pipelines, documentation, and thesis-related result artifacts. Its existing `results/framework/real_hardware_validation/` folder should **not automatically be treated as the raw IBM Marrakesh records for Beyond Parity**.

To independently verify the manuscript's specific hardware experiment, a future approved research release should provide, where permitted:

1. Exact experiment scripts, pinned dependencies, seed list, dataset split logic, and fixed circuit parameter records.
2. Transpiled/frozen circuit definitions, backend identification and execution dates, and non-sensitive job identifiers.
3. Raw or appropriately sanitized X/Y/Z measurement counts, shots, and circuit-to-sample mapping.
4. Decoder configurations, analysis scripts, and per-seed predictions/metrics used to regenerate the tables and confidence intervals.
5. Checksums, execution instructions, and a mapping between the released artifacts and manuscript tables/figures.

**Not currently claimed here:** that those study-specific records are publicly released, that this repository independently reproduces all Beyond Parity tables, or that manuscript results were generated by the thesis pipelines without modification.

## Source and attribution

This document summarizes the submitted *Beyond Parity* manuscript (Jesus Gil and David Hart). The manuscript's author-contribution statement attributes study conception, experimental framework, simulation and hardware experiments, statistical analysis, and original draft to Jesus Gil, with methodological guidance and substantive manuscript review from Dr. David Hart.

**Sharing:** Release study-specific raw artifacts or the complete manuscript only after checking coauthor permissions and applicable journal-sharing requirements.
