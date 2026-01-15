# AGENT GUIDE & MEMORY

This document serves as a persistent context and rulebook for the AI Agent working on the **ML_ZULF** project. It should be updated regularly to reflect architectural decisions, constraints, and project status.

## 1. Project Overview
**Name**: ML_ZULF (Machine Learning Driven Zero-Ultra-Low-Field NMR Optimizer)
**Goal**: Optimize simulation parameters (J-couplings, T2, Window functions) to match experimental ZULF-NMR spectra using Random Walk code.
**Core Dependency**: Relies on `spinach_bridge` to interface with MATLAB-based Spinach simulation kernel found in `references/ZULF_NMR_Suite`.

## 2. Architectural Constraints (CRITICAL)
*   **UI Framework**: All new UI components **MUST** use **PySide6**.
*   **Design Pattern**: **Strictly follow** the implementation style of the reference project (`references/ZULF_NMR_Suite`).
    *   **Threading**: Use `QThread` workers for long-running tasks (optimization/simulation). NEVER block the main GUI thread.
    *   **Communication**: Use `Signal` and `Slot` for Worker <-> UI communication.
    *   **Layout**: Use `QMainWindow`, `QWidget` with `QVBoxLayout/QHBoxLayout`, and standard Qt widgets.
    *   **Plotting**: Embed `matplotlib` using `FigureCanvasQTAgg`.
*   **Data Structures**:
    *   **Spectrum**: CSV with typically 2 columns (Frequency, Amplitude) or complex data.
    *   **Molecule**: CSV structure (Row 0: Isotopes, Row 1+: J-coupling matrix).
    *   **Config**: `setting.json` for experimental parameters.

## 3. Code Structure & Modules
*   `main.py`: **Dual Entry Point**.
    *   No args -> Launches UI (`run_ui()`).
    *   `--cli` -> Runs CLI mode (`run_cli()`).
    *   **DO NOT** clutter this file with logic. Import from `src`.
*   `src/core/optimizer.py`: Contains `ZulfOptimizer`.
    *   Logic is separated from UI.
    *   Uses `callback` function for progress reporting.
*   `src/ui/`: Contains all PySide6 code.
    *   `optimization_window.py`: Main GUI and Worker classes.
*   `src/utils/`: Utility package (Must contain `__init__.py`).
    *   `loaders.py`: Centralized data loading logic.
*   `references/`: Contains the legacy/reference code. **Read-only** generally, unless debugging the bridge.

## 4. Current Status (As of Jan 14, 2026)
*   [x] Backend: Random Walk Optimizer operational.
*   [x] Integration: MATLAB/Spinach bridge connected via `simulation_wrapper.py`.
*   [x] UI: Basic PySide6 Interface implemented (`OptimizationWindow`).
    *   Supports loading Experiment (.csv) and Molecule (.csv).
    *   Runs optimization in background thread.
    *   Visualizes real-time progress (Log).
    *   *Pending*: Real-time plotting hookup (Worker emits signal, need to ensure thread-safe plotting).
*   [x] Refactoring: `src.utils` fixed to be a proper package. `main.py` cleaned up.

## 5. Operational Rules
1.  **Git**: Always `git status` before committing. Keep commits atomic and descriptive.
2.  **File Editing**: When using `replace_string_in_file`, always provide 3-5 lines of context.
3.  **Environment**: Python 3.x. Requires `matlab` engine (or mock fallback if unavailable).
4.  **Language & Style**: **NO Chinese characters** (comments or code) and **NO Emojis** in source files. Use English for all code and comments.

## 6. Todo / Next Steps
*   **Plotting**: Connect the `new_best` signal in `OptimizationWindow` to update the Matplotlib canvas with the simulated spectrum of the best parameters found so far.
*   **Validation**: Add robust error handling if MATLAB engine fails mid-optimization.
*   **Config**: Allow editing optimization constraints (min/max J values) from the UI.
