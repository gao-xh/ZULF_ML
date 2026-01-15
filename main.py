import sys
import os
import argparse
import numpy as np
from PySide6.QtWidgets import QApplication

# Local Imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from src.core.optimizer import ZulfOptimizer
from src.utils.loaders import load_experimental_and_config, load_molecule_from_csv
from src.ui.optimization_window import OptimizationWindow

def run_cli(args):
    """Run optimization in CLI mode."""
    print("--- ZULF-NMR Optimizer (CLI) ---")
    
    # 1. Load Data
    if args.data and os.path.exists(args.data):
        try:
            exp_spectrum, sampling_rate, _, _ = load_experimental_and_config(args.data)
            print(f"Loaded Experimental Data: {args.data}")
            exp_fid = None
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    else:
        print("Error: Please provide a valid data folder path using --data")
        return

    # 2. Load Molecule
    spins = ['1H', '13C'] # Default
    init_j = np.zeros((2,2)) 
    
    if args.molecule and os.path.exists(args.molecule):
        try:
            spins, init_j = load_molecule_from_csv(args.molecule)
            print(f"Loaded Molecule: {spins}")
        except Exception as e:
            print(f"Error loading molecule: {e}")
            return
    else:
        print("Warning: No molecule file provided, using default/placeholders.")

    # 3. Setup Optimizer
    optimizer = ZulfOptimizer(
        spins=spins,
        sampling_rate=sampling_rate,
        exp_spectrum=exp_spectrum
    )
    
    # 4. Run
    print(f"Starting optimization ({args.iter} iterations)...")
    init_sg = 5
    init_trunc = 1000
    init_t2 = 1.0
    
    # Run
    optimizer.config.max_iterations = args.iter
    best_params, history = optimizer.run(init_j, init_sg, init_trunc, init_t2)
    
    # Result
    print("\n--- Optimization Complete ---")
    print(f"Best Cost: {history[-1]:.4f}")
    print(f"Best J:\n{best_params[0]}")
    
    if args.output:
        optimizer.plot_comparison(save_path=args.output)
        print(f"Plot saved to {args.output}")

def run_ui():
    """Launch the Graphical User Interface."""
    print("Launching UI...")
    app = QApplication(sys.argv)
    window = OptimizationWindow()
    window.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="ZULF-NMR ML Optimizer")
    
    # Modes
    parser.add_argument('--ui', action='store_true', help="Launch the GUI (Default if no other args provided)")
    parser.add_argument('--cli', action='store_true', help="Run in CLI mode")
    
    # CLI Arguments
    parser.add_argument('--data', type=str, help="Path to experimental data folder (containing spectrum.csv)")
    parser.add_argument('--molecule', type=str, help="Path to molecule structure.csv")
    parser.add_argument('--iter', type=int, default=100, help="Number of iterations")
    parser.add_argument('--output', type=str, default="result.png", help="Path to save result plot")

    args = parser.parse_args()

    # Decision Logic
    if args.cli:
        run_cli(args)
    elif args.ui:
        run_ui()
    else:
        # Default behavior: If no specific CLI args, launch UI?
        # Or check if any functional args are present.
        if args.data or args.molecule:
            # User tried to run CLI but forgot --cli flag, let's just run it
            run_cli(args)
        else:
            # Default to UI
            run_ui()

if __name__ == "__main__":
    main()
