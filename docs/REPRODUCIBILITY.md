# Reproducibility Guide

This document describes the supported workflow for setting up, validating, and running the Noise-Robust HQNN research codebase.

## 1. Python Environment

The project targets Python 3.11.

```bash
conda create -n hqnn python=3.11 -y
conda activate hqnn
pip install -r env/requirements.txt
```

For development and automated testing:

```bash
pip install -r env/requirements-dev.txt
```

`requirements-dev.txt` includes the research dependencies plus the test tooling used by CI.

## 2. Fast Verification

Run the deterministic unit-test suite from the repository root:

```bash
pytest -q tests
```

The current lightweight suite verifies:

- robustness metric calculations;
- aggregate noise-summary behavior;
- synthetic dataset shape and binary labels;
- deterministic preprocessing under a fixed random seed;
- binary Iris preprocessing; and
- WDBC dimensionality after PCA-based reduction.

These tests intentionally avoid expensive quantum optimization or full experiment reruns so that they remain appropriate for continuous integration.

## 3. Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/tests.yml
```

The workflow runs on pushes and pull requests targeting `main`, creates a Python 3.11 environment, installs `env/requirements-dev.txt`, and runs the unit tests.

## 4. Run the Framework

From the repository root:

```bash
python run_framework.py
```

Individual research pipelines can also be run directly, for example:

```bash
PYTHONPATH=. python pipelines/main_learned_readout_hqnn.py
PYTHONPATH=. python pipelines/main_multi_observable_hqnn.py
PYTHONPATH=. python pipelines/main_architecture_search_hqnn.py
PYTHONPATH=. python pipelines/main_best_architecture_noise_sweep.py
PYTHONPATH=. python pipelines/main_statistical_validation.py
```

Because many experiments are stochastic, reported values should be interpreted together with their random seeds, shot counts, circuit configuration, noise settings, dataset split, and classical model configuration.

## 5. Docker Environment

A Python 3.11 Docker environment is provided in `docker/Dockerfile`.

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t noise-robust-hqnn .
```

Start an interactive container while mounting the repository:

```bash
docker run --rm -it -v "$PWD:/workspace" noise-robust-hqnn
```

Inside the container, the repository is available at `/workspace`.

## 6. Research Results

Generated outputs include JSON summaries, CSV summaries, plots, noise curves, heatmaps, and statistical reports. The repository's reported results are experimental outputs from specific configurations and are not universal performance guarantees or claims of general quantum advantage.

For publication-level reproduction, preserve the exact environment, configuration values, random seeds, and generated output files associated with the reported experiment.

## 7. Citation

Machine-readable citation metadata is available in `CITATION.cff`.

When a thesis-derived paper or archival software release becomes available, the citation metadata should be updated with the final publication identifier, release version, and DOI if applicable.
