# Framework Contributions

This repository uses Qiskit, Qiskit Aer, Qiskit Machine Learning, Cirq, PennyLane, and scikit-learn as computational libraries.

The contribution of this project is the reusable benchmarking framework built around those libraries.

## What the Framework Adds

1. Standardized dataset layer  
   Code: `framework/datasets.py`

2. Reusable noise-analysis toolbox  
   Code: `framework/noise_channels.py`

3. Robustness metrics not provided as one unified workflow in default packages  
   Code: `framework/robustness_metrics.py`

4. Standardized reporting layer  
   Code: `framework/reporting.py`

5. Benchmark orchestration layer  
   Code: `framework/benchmark_runner.py`

6. Main contribution pipelines  
   Code: `pipelines/`

7. Domain-inspired cybersecurity anomaly benchmark  
   Code: `demos/industry/demo11_cyber_anomaly_qiskit.py`

## Nonstandard Framework Features

- `accuracy_drop`
- `robustness_score`
- `degradation_slope`
- `training_instability`
- `cross_framework_deviation`
- framework-level CSV/JSON outputs
- cross-framework validation summaries
- hybrid-versus-classical benchmark summaries

## Why the Demos Are Short

The demos are intentionally short because reusable logic has been moved into the framework layer.

The demos now function as experimental instances of the framework rather than isolated scripts.
