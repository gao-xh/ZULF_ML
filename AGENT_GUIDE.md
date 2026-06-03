# AGENT GUIDE

This document is the working guide for the active ML_ZULF codebase.

## 1. Project Direction

**Name**: ML_ZULF

**Current objective**: fit simulated ZULF-NMR spectra to experimental data using staged black-box optimization.

**Immediate architectural direction**:
- keep the existing UI and CLI entry points
- stop treating the project as a generic ML scaffold
- move from single expensive optimization loop to staged optimization
- introduce a fast Python backend for coarse search
- retain MATLAB/Spinach or equivalent full simulation for final refinement

## 2. Source of Truth

The active application is defined by these files:

- `main.py`
- `src/core/optimizer.py`
- `src/core/simulation_wrapper.py`
- `src/core/cost.py`
- `src/processing/signal.py`
- `src/ui/optimization_window.py`
- `src/utils/loaders.py`

The `references/` directory is retained intentionally and should remain available for extraction, comparison, and validation.

## 3. Architectural Constraints

- UI must remain in PySide6.
- Long-running optimization and simulation work must stay off the main GUI thread.
- UI must orchestrate work, not contain optimization or simulation logic.
- Simulation backends must be swappable behind a narrow interface.
- Experimental data handling should move toward a unified data object instead of loose tuples.

## 4. Target Backend Strategy

The intended optimizer design is staged:

1. Stage A: coarse search with an eigenvalue-based backend
   - optimize mainly J-related physical parameters
   - score primarily on peak-position error

2. Stage B: intermediate ranking with a broadened-stick or lightweight Python spectrum backend
   - introduce T2 and coarse shape terms

3. Stage C: full refinement with a high-fidelity backend
   - use full-spectrum comparison and final processing parameters

Two backend roles should exist in the active codebase:

- `FastSpectrumBackend`: cheap, approximate, suitable for global search
- `FullSpectrumBackend`: slower, higher fidelity, suitable for final ranking and validation

## 5. Practical Rules

- Keep `references/` intact.
- Prefer extraction and adaptation over direct copying of large legacy files.
- Remove unused scaffold code from the active app when confirmed unreferenced.
- Keep source code comments and identifiers in English.
- Keep documentation aligned with the active architecture.

## 6. Current Cleanup State

Completed in this cleanup pass:

- removed unused generic ML scaffold files from `src/`
- removed duplicate UI launcher and obsolete backup file
- rewrote top-level documentation to reflect the active project

## 7. Next Implementation Priorities

1. Define simulation backend interfaces.
2. Extract a minimal Python 1D simulation backend from the reference kernel.
3. Split optimization into staged objectives and staged backends.
4. Add checkpointing and resume support.
5. Add narrow tests for loaders, cost functions, and backend agreement.
