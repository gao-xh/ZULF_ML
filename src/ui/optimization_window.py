import sys
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QFormLayout, 
    QGroupBox, QComboBox, QTextEdit, QFileDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.core.optimizer import ZulfOptimizer
from src.config import OptimizerConfig
from src.core.simulation_wrapper import ZulfSimulation
from src.utils.loaders import load_experimental_and_config

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

    def __init__(self, optimizer, init_params):
        super().__init__()
        self.optimizer = optimizer
        self.init_params = init_params
        self._is_running = True

    def run(self):
        try:
            self.log.emit("Starting optimization...")
            # Unpack init_params: init_j, sg, trunc, t2
            # Run optimizer with callback
            best_params, history = self.optimizer.run(
                *self.init_params, 
                callback=self._step_callback
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
        self.lbl_mol_status = QLabel("No molecule loaded")
        
        data_form.addRow(self.btn_load_exp, self.lbl_exp_status)
        data_form.addRow(self.btn_load_mol, self.lbl_mol_status)
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

        param_form.addRow("Max Iterations:", self.spin_steps)
        param_form.addRow("Plot Interval:", self.spin_plot_interval)
        param_form.addRow("Linewidth (Hz):", self.spin_t2)
        param_form.addRow("SG Window (pts):", self.spin_sg)
        param_form.addRow("Truncation (pts):", self.spin_trunc)
        
        param_group.setLayout(param_form)
        settings_layout.addWidget(param_group)
        
        # 3. Control Buttons
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Optimization")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
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
        
        # Matplotlib Canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Optimization Progress")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Intensity")
        
        # Connect Signals
        self.btn_load_exp.clicked.connect(self.load_experiment)
        self.btn_load_mol.clicked.connect(self.load_molecule)
        self.btn_start.clicked.connect(self.start_optimization)
        self.btn_stop.clicked.connect(self.stop_optimization)

        # Internal State
        self.exp_spectrum = None # (freq, amp)
        self.exp_fid = None      # array or None
        self.sampling_rate = None
        self.j_coupling = None
        self.loaded_config = None

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
                self.ax.clear()
                self.ax.plot(self.exp_spectrum[0], self.exp_spectrum[1], 'k-', alpha=0.5, label='Experiment')
                self.ax.legend()
                self.ax.set_title("Experimental Data")
                self.canvas.draw()
                
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
                 self.lbl_mol_status.setText(f"Loaded: {len(self.isotopes)} spins")
                 self.log(f"Loaded molecule from {path}")
             except Exception as e:
                 self.log(f"Error loading molecule: {e}")

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
            
            # 5. Start Worker
            self.worker = OptimizationWorker(self.optimizer, init_params)
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
        
        # Real-time Plotting
        if viz_data:
            try:
                self.ax.clear()
                
                # Extract Data
                sim_freq = viz_data['sim_freq']
                sim_amp = viz_data['sim_amp']
                exp_freq = viz_data['exp_freq']
                exp_amp = viz_data['exp_amp']
                
                # Plot Experimental (Black)
                self.ax.plot(exp_freq, exp_amp, 'k-', alpha=0.6, label='Experiment')
                
                # Plot Simulated (Red Dashed)
                # Auto-scaling if sim amplitude is arbitrary vs exp
                if np.max(sim_amp) > 0 and np.max(exp_amp) > 0:
                    scale = np.max(exp_amp) / np.max(sim_amp)
                    self.ax.plot(sim_freq, sim_amp * scale, 'r--', label='Simulated (Best)')
                else:
                    self.ax.plot(sim_freq, sim_amp, 'r--', label='Simulated (Best)')
                    
                self.ax.set_title(f"Optimization Progress (Iter {iteration}, Cost {cost:.2f})")
                self.ax.set_xlabel("Frequency (Hz)")
                self.ax.legend()
                self.canvas.draw()
                
            except Exception as e:
                print(f"Plotting error: {e}")

    def on_finished(self, best_params, history):
        self.log("Optimization Finished.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.worker = None

    def on_failed(self, error):
        self.log(f"Optimization Failed: {error}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.worker = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationWindow()
    window.show()
    sys.exit(app.exec())
