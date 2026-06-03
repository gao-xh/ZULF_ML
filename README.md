# ML_ZULF

ML_ZULF is a ZULF-NMR parameter optimization project built around three layers:

- experimental data loading and processing
- spectrum simulation backends
- staged parameter optimization with a PySide6 UI and CLI entry point

The active code path is no longer a generic machine learning scaffold. The repository now focuses on fitting simulated spectra to experimental ZULF-NMR data.

## Current Goal

The current development direction is to replace single-stage random walk with a staged optimizer:

1. fast coarse search using a Python eigenspectrum backend
2. intermediate ranking using broadened stick spectra
3. final refinement using a full-spectrum backend such as Spinach or Python FID simulation

## Active Entry Points

- `main.py`: main application entry point
  - no args: launch the PySide6 UI
  - `--cli`: run command-line optimization
- `references/`: retained as the source of legacy code, architecture examples, and simulation kernels

## Active Source Layout

- `src/core/optimizer.py`: current optimization loop and evaluation orchestration
- `src/core/simulation_wrapper.py`: current MATLAB/Spinach-backed simulation wrapper
- `src/core/cost.py`: spectrum comparison and cost calculation
- `src/processing/signal.py`: experimental signal processing utilities
- `src/ui/optimization_window.py`: PySide6 UI and worker thread
- `src/utils/loaders.py`: experiment and molecule loading

## Planned Architecture

The target architecture is a dual-backend design:

- `FastSpectrumBackend`: eigenvalue-based coarse search backend
- `FullSpectrumBackend`: high-fidelity backend for refinement

The optimizer should choose backend and objective by stage instead of using one expensive pipeline for every iteration.

## Repository Policy

- keep `references/` intact for comparison and extraction
- remove unused scaffold files from the active app
- keep UI code in PySide6
- keep optimization logic out of the UI layer

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the application:

```bash
python main.py
```

3. For current planning status, see `todo.txt`.
