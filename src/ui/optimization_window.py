import sys
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QFormLayout, 
    QGroupBox, QComboBox, QTextEdit, QFileDialog, QInputDialog
)

from src.core.optimizer import ZulfOptimizer
from src.config import OptimizerConfig
from src.core.simulation_wrapper import ZulfSimulation
from src.utils.loaders import load_experimental_and_config
from src.ui.molecule_editor import JCouplingEditorDialog, parse_isotopes
from src.ui.plotting import SpectrumWidget

# ---------- Worker Thread ----------
class OptimizationWorker(QThread):
    """
    Worker thread that runs the ZulfOptimizer loop.
    Emits signals for progress updates and completion.
    """
    log = Signal(str)
    progress = Signal(int, float)            # iteration, current_cost
    new_best = Signal(int, float, object, object)    # iteration, best_cost, best_params, viz_data
    finished = Signal(object, list)          # best_params, history
    failed = Signal(str)

    def __init__(self, optimizer, init_params, freq_range=None, variable_config=None):
        super().__init__()
        self.optimizer = optimizer
        self.init_params = init_params
        self.freq_range = freq_range
        self.variable_config = variable_config
        self._is_running = True

    def run(self):
        try:
            self.log.emit("Starting optimization...")
            # Unpack init_params: init_j, sg, trunc, t2
            # Run optimizer with callback
            best_params, history = self.optimizer.run(
                *self.init_params, 
                callback=self._step_callback,
                freq_range=self.freq_range,
                variable_config=self.variable_config
            )
            self.finished.emit(best_params, history)
        except Exception as e:
            self.failed.emit(str(e))

    def stop(self):
        self._is_running = False

    def _step_callback(self, i, curr_cost, best_cost, best_params, viz_data=None):
        if not self._is_running:
            return False # Stop optimizer
        
        self.progress.emit(i, curr_cost)
        
        # Emit new best if applicable (cost has improved)
        if best_cost == curr_cost and viz_data is not None: 
             self.new_best.emit(i, best_cost, best_params, viz_data)
             
        return True


# ---------- Main Window ----------
class OptimizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZULF-NMR Parameter Optimizer")
        self.resize(1200, 800)
        
        # Backend objects
        self.optimizer = None
        self.worker = None
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Left Panel: Settings ---
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_panel.setMaximumWidth(400)
        main_layout.addWidget(settings_panel)
        
        # 1. Data Loader Group
        data_group = QGroupBox("Data & Configuration")
        data_form = QFormLayout()
        
        self.btn_load_exp = QPushButton("Load Experiment Data")
        self.lbl_exp_status = QLabel("No data loaded")
        self.btn_load_mol = QPushButton("Load Molecule Structure")
        self.btn_top_sys = QPushButton("Define System Manually")  # New Button
        self.lbl_mol_status = QLabel("No molecule loaded")
        
        data_form.addRow(self.btn_load_exp, self.lbl_exp_status)
        data_form.addRow(self.btn_load_mol, self.lbl_mol_status)
        data_form.addRow(self.btn_top_sys, QLabel("(Or build manually)"))
        data_group.setLayout(data_form)
        settings_layout.addWidget(data_group)
        
        # 2. Optimization Parameters
        param_group = QGroupBox("Optimization Settings")
        param_form = QFormLayout()
        
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(10, 10000)
        self.spin_steps.setValue(100)
        
        self.spin_plot_interval = QSpinBox()
        self.spin_plot_interval.setRange(1, 1000)
        self.spin_plot_interval.setValue(10)

        self.spin_plot_interval.setToolTip("Update plot every N steps")

        # --- New Parameter Inputs ---
        self.spin_t2 = QDoubleSpinBox()
        self.spin_t2.setRange(0.01, 20.0)
        self.spin_t2.setValue(0.5)
        self.spin_t2.setSingleStep(0.1)

        self.spin_sg = QSpinBox()
        self.spin_sg.setRange(5, 501)
        self.spin_sg.setValue(101)
        self.spin_sg.setSingleStep(2)

        self.spin_trunc = QSpinBox()
        self.spin_trunc.setRange(0, 5000)
        self.spin_trunc.setValue(160)
        self.spin_trunc.setSingleStep(10)

        # --- Frequency Range Inputs ---
        self.spin_freq_min = QDoubleSpinBox()
        self.spin_freq_min.setRange(0, 10000)
        self.spin_freq_min.setValue(0)
        self.spin_freq_min.setSingleStep(10)
        self.spin_freq_min.setToolTip("Minimum frequency for cost calculation (Hz)")

        self.spin_freq_max = QDoubleSpinBox()
        self.spin_freq_max.setRange(0, 10000)
        self.spin_freq_max.setValue(400) # Default Zulf range
        self.spin_freq_max.setSingleStep(10)
        self.spin_freq_max.setToolTip("Maximum frequency for cost calculation (Hz)")

        param_form.addRow("Max Iterations:", self.spin_steps)
        param_form.addRow("Plot Interval:", self.spin_plot_interval)
        param_form.addRow("Linewidth (Hz):", self.spin_t2)
        param_form.addRow("SG Window (pts):", self.spin_sg)
        param_form.addRow("Truncation (pts):", self.spin_trunc)
        param_form.addRow("Freq Min (Hz):", self.spin_freq_min)
        param_form.addRow("Freq Max (Hz):", self.spin_freq_max)
        
        param_group.setLayout(param_form)
        settings_layout.addWidget(param_group)
        
        # 3. Control Buttons
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Optimization")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("Save Report")
        self.btn_save.setEnabled(False)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_save)
        settings_layout.addLayout(control_layout)
        
        # 4. Log
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        settings_layout.addWidget(QLabel("Log:"))
        settings_layout.addWidget(self.text_log)
        
        # --- Right Panel: Visualization ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        main_layout.addWidget(plot_panel)
        
        # Use Custom SpectrumWidget
        self.plot_widget = SpectrumWidget(self)
        plot_layout.addWidget(self.plot_widget)
        
        # Connect Signals
        self.btn_load_exp.clicked.connect(self.load_experiment)
        self.btn_load_mol.clicked.connect(self.load_molecule)
        self.btn_top_sys.clicked.connect(self.open_system_builder)
        self.btn_start.clicked.connect(self.start_optimization)
        self.btn_stop.clicked.connect(self.stop_optimization)
        self.btn_save.clicked.connect(self.save_results)

        # Internal State
        self.exp_spectrum = None # (freq, amp)
        self.exp_fid = None      # array or None
        self.sampling_rate = None
        self.j_coupling = None
        self.variable_config = None # New: Dict for variable params
        self.loaded_config = None
        self.best_viz_data = None
        self.best_params_result = None
        self.best_cost_result = None

    def log(self, message):
        self.text_log.append(message)

    def load_experiment(self):
        # We need to select a FOLDER now, because of NMRduino format (multiple files)
        # or legacy format (spectrum.csv + setting.json)
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Data Folder")
        if folder:
            try:
                (spectrum, fid), sr, _, _ = load_experimental_and_config(folder)
                
                self.exp_spectrum = spectrum
                self.exp_fid = fid
                self.sampling_rate = sr
                
                sp_len = len(spectrum[0]) if spectrum else 0
                fid_len = len(fid) if fid is not None else 0
                
                self.lbl_exp_status.setText(f"Spectrum: {sp_len} pts | FID: {fid_len} pts")
                self.log(f"Loaded data from {folder}")
                self.log(f"Sampling Rate: {sr} Hz")
                
                # Plot
                self.plot_widget.update_plot(self.exp_spectrum[0], self.exp_spectrum[1])
                
            except Exception as e:
                self.log(f"Error loading data: {e}")

    def load_molecule(self):
         path, _ = QFileDialog.getOpenFileName(self, "Open Structure CSV", "", "CSV Files (*.csv)")
         if path:
             try:
                 import csv
                 with open(path, 'r', encoding='utf-8') as f:
                     reader = csv.reader(f)
                     rows = list(reader)
                 if not rows: raise ValueError("Empty File")
                 
                 # Row 0: Isotopes
                 self.isotopes = [iso.strip() for iso in rows[0] if iso.strip()]
                 
                 # Rows 1+: J Matrix
                 j_data = []
                 for row in rows[1:]:
                     vals = [float(x) for x in row if x.strip()]
                     if vals: j_data.append(vals)
                 
                 self.j_coupling = np.array(j_data)
                 self.variable_config = None # Reset params
                 
                 self.lbl_mol_status.setText(f"Loaded: {len(self.isotopes)} spins")
                 self.log(f"Loaded molecule from {path}. Mode: Numeric")
             except Exception as e:
                 self.log(f"Error loading molecule: {e}")

    def open_system_builder(self):
        """Open the manual J-Coupling Editor dialog."""
        # 1. Ask for Isotopes
        text, ok = QInputDialog.getText(
            self, "Define System", 
            "Enter isotopes (comma separated, e.g. 1H, 13C, 1H):"
        )
        if not ok or not text.strip():
            return
            
        isotopes = parse_isotopes(text)
        if not isotopes:
            self.log("No valid isotopes entered.")
            return
            
        # 2. Open Editor
        # Pass current j_coupling if size matches, else None
        current_j = None
        if self.j_coupling is not None and self.j_coupling.shape[0] == len(isotopes):
            current_j = self.j_coupling
            
        dlg = JCouplingEditorDialog(isotopes, current_j, self)
        if dlg.exec():
            # 3. Retrieve Result
            self.isotopes = isotopes
            self.j_coupling = dlg.result_matrix
            self.variable_config = dlg.variable_config
            
            n_spins = len(self.isotopes)
            mode = "Variable Mode" if self.variable_config else "Numeric Mode"
            self.lbl_mol_status.setText(f"Manual: {n_spins} spins ({mode})")
            self.log(f"System defined manually: {isotopes}")
            self.log(f"J-Coupling Matrix updated ({n_spins}x{n_spins}). Mode: {mode}")

    def start_optimization(self):
        if self.exp_spectrum is None or self.j_coupling is None:
            self.log("Error: Please load both Experiment Data and Molecule Structure.")
            return

        # Initialize Optimizer
        try:
            # 1. Config
            config = OptimizerConfig()
            config.max_iterations = self.spin_steps.value()
            config.plot_interval = self.spin_plot_interval.value()

            # Map UI params to Config (for elasticity centers)
            config.t2_linewidth.initial_value = self.spin_t2.value()
            config.sg_window.initial_value = self.spin_sg.value()
            config.truncation.initial_value = self.spin_trunc.value()
            
            # 2. Parameters
            # Default spins if not set (fallback)
            spins = getattr(self, 'isotopes', ['1H'] * self.j_coupling.shape[0])
            
            # Estimate Sampling Rate (Sweep)
            # Use loaded sampling rate if available
            sr = self.sampling_rate if self.sampling_rate else 400.0
            
            # 3. Instantiate Optimizer
            try:
                self.optimizer = ZulfOptimizer(
                    spins=spins,
                    sampling_rate=sr,
                    exp_spectrum=self.exp_spectrum,
                    exp_fid=self.exp_fid
                )
            except Exception as e:
                self.log(f"Optimizer instantiation failed: {e}")
                return
            
            # Inject Config
            self.optimizer.config = config
            
            # 4. Initial Parameters Tuple
            # (j_coupling_matrix, sg_window, trunc, t2)
            init_params = (
                self.j_coupling,
                config.sg_window.initial_value,
                config.truncation.initial_value,
                config.t2_linewidth.initial_value
            )
            
            # Get Frequency Range
            f_min = self.spin_freq_min.value()
            f_max = self.spin_freq_max.value()
            freq_range = (f_min, f_max)

            # 5. Start Worker
            # Pass variable_config if available
            var_config = getattr(self, 'variable_config', None)
            
            self.worker = OptimizationWorker(self.optimizer, init_params, freq_range, var_config)
            self.worker.log.connect(self.log)
            self.worker.progress.connect(self.on_progress)
            self.worker.new_best.connect(self.on_new_best)
            self.worker.finished.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            
            self.worker.start()
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.log("Optimization started...")
            
        except Exception as e:
            self.log(f"Failed to start: {e}")

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stopping...")

    def on_progress(self, iteration, cost):
        # Update progress bar or log occasionally
        if iteration % 10 == 0:
            self.log(f"Iter {iteration}: Cost {cost:.4f}")

    def on_new_best(self, iteration, cost, params, viz_data):
        self.log(f"New Best found at iter {iteration} (Cost: {cost:.4f})")
        
        # Store for saving later
        self.best_viz_data = viz_data
        self.best_params_result = params
        self.best_cost_result = cost
        
        # Real-time Plotting
        if viz_data:
            try:
                # Extract Data
                sim_freq = viz_data['sim_freq']
                sim_amp = viz_data['sim_amp']
                exp_freq = viz_data['exp_freq']
                exp_amp = viz_data['exp_amp']
                
                # Update Custom Widget
                self.plot_widget.update_plot(
                    exp_freq, exp_amp, 
                    sim_freq, sim_amp, 
                    cost=cost, iter_num=iteration
                )
                
            except Exception as e:
                print(f"Plotting error: {e}")

    def on_finished(self, best_params, history):
        self.log("Optimization Finished.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.worker = None

    def on_failed(self, error):
        self.log(f"Optimization Failed: {error}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Still allow saving if we found *some* result before failure
        if self.best_params_result is not None:
             self.btn_save.setEnabled(True)
        self.worker = None

    def save_results(self):
        """Save the optimization report to a CSV/JSON file."""
        if not self.best_viz_data or not self.best_params_result:
            self.log("No results to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Optimization Report", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if not path:
            return

        try:
            # Unpack Params
            final_j, final_sg, final_trunc, final_t2 = self.best_params_result
            
            # Prepare Data
            sim_freq = self.best_viz_data['sim_freq']
            sim_amp = self.best_viz_data['sim_amp']
            exp_freq = self.best_viz_data['exp_freq']
            exp_amp = self.best_viz_data['exp_amp']
            
            # 1. Save as CSV (Spectra + Header Metadata)
            if path.lower().endswith('.csv'):
                import csv
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header Info
                    writer.writerow(["# Optimization Report"])
                    writer.writerow(["# Final Cost", self.best_cost_result])
                    writer.writerow(["# T2 Linewidth (Hz)", final_t2])
                    writer.writerow(["# SG Window", final_sg])
                    writer.writerow(["# Truncation", final_trunc])
                    writer.writerow([])
                    
                    # J-Coupling Section
                    writer.writerow(["# J-Coupling Matrix (Flattened or Rows)"])
                    # Simple dump of matrix rows
                    for row in final_j:
                        writer.writerow(["# J_Row"] + list(row))
                    writer.writerow([])
                    
                    # Spectral Data
                    writer.writerow(["Frequency (Hz)", "Experimental Amp", "Simulated Amp"])
                    
                    # Ensure lengths match (interpolate if needed, but usually they are on same grid if optimizer did its job)
                    # In optimizer.py: sim is generated with npoints=len(exp) and sweep=max(exp).
                    # So they should be 1:1.
                    min_len = min(len(exp_freq), len(sim_freq))
                    
                    # Scale Simulation for report (visual match)
                    scale = 1.0
                    if np.max(sim_amp) > 0 and np.max(exp_amp) > 0:
                        scale = np.max(exp_amp) / np.max(sim_amp)
                    
                    for i in range(min_len):
                        writer.writerow([
                            exp_freq[i],
                            exp_amp[i],
                            sim_amp[i] * scale
                        ])
                        
                self.log(f"Saved CSV report to {path}")

            # 2. Save as JSON
            elif path.lower().endswith('.json'):
                import json
                report = {
                    "final_cost": float(self.best_cost_result),
                    "parameters": {
                        "t2_linewidth": float(final_t2),
                        "sg_window": int(final_sg),
                        "truncation": int(final_trunc),
                        "j_coupling": final_j.tolist()
                    },
                    "spectra": {
                         "frequency": exp_freq.tolist(),
                         "experimental": exp_amp.tolist(),
                         "simulated": (sim_amp * (np.max(exp_amp)/np.max(sim_amp) if np.max(sim_amp) > 0 else 1.0)).tolist()
                    }
                }
                with open(path, 'w') as f:
                    json.dump(report, f, indent=4)
                self.log(f"Saved JSON report to {path}")

        except Exception as e:
            self.log(f"Error saving results: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationWindow()
    window.show()
    sys.exit(app.exec())
