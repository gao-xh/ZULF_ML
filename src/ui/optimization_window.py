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

# ---------- Worker Thread ----------
class OptimizationWorker(QThread):
    """
    Worker thread that runs the ZulfOptimizer loop.
    Emits signals for progress updates and completion.
    """
    log = Signal(str)
    progress = Signal(int, float)            # iteration, current_cost
    new_best = Signal(int, float, object)    # iteration, best_cost, best_params
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

    def _step_callback(self, i, curr_cost, best_cost, best_params):
        if not self._is_running:
            return False # Stop optimizer
        
        self.progress.emit(i, curr_cost)
        
        # Check if this is a new best locally to emit specific signal?
        # The optimizer prints it, but we want UI to know.
        # We can just emit 'new_best' every time or track it here.
        # For now, let's emit if it matches the optimizer's best
        if best_cost == curr_cost: 
             self.new_best.emit(i, best_cost, best_params)
             
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

        param_form.addRow("Max Iterations:", self.spin_steps)
        param_form.addRow("Plot Interval:", self.spin_plot_interval)
        
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
        self.exp_data = None  # (freq, spec)
        self.j_coupling = None
        self.loaded_config = None

    def log(self, message):
        self.text_log.append(message)

    def load_experiment(self):
        # Placeholder for loading logic (csv/json)
        # Using QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Open Spectrum CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                # Load logic here or use main.py helper
                data = np.loadtxt(path, delimiter=',')
                if data.shape[1] < 2:
                    raise ValueError("CSV must have at least 2 columns")
                self.exp_data = (data[:,0], data[:,1])
                self.lbl_exp_status.setText(f"Loaded: {len(self.exp_data[0])} pts")
                self.log(f"Loaded experiment data from {path}")
                
                # Initial plot
                self.ax.plot(self.exp_data[0], self.exp_data[1], 'k-', alpha=0.5, label='Experiment')
                self.ax.legend()
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
        if self.exp_data is None or self.j_coupling is None:
            self.log("Error: Please load both Experiment Data and Molecule Structure.")
            return

        # Initialize Optimizer
        try:
            # 1. Config
            config = OptimizerConfig()
            config.max_iterations = self.spin_steps.value()
            config.plot_interval = self.spin_plot_interval.value()
            
            # 2. Parameters
            # Default spins if not set (fallback)
            spins = getattr(self, 'isotopes', ['1H'] * self.j_coupling.shape[0])
            
            # Estimate Sampling Rate (Sweep)
            exp_freq, exp_amp = self.exp_data
            sweep = float(np.max(exp_freq)) if len(exp_freq) > 0 else 400.0
            
            # 3. Instantiate Optimizer
            try:
                self.optimizer = ZulfOptimizer(
                    spins=spins,
                    sampling_rate=sweep,
                    exp_spectrum=self.exp_data
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

    def on_new_best(self, iteration, cost, params):
        self.log(f"Checking new best at iter {iteration} (Cost: {cost:.4f})")
        # Trigger plotting
        # We need to simulate spectrum with these params to plot it
        # Prefer doing this in main thread? Or have worker output the spec?
        # ZulfOptimizer.simulate_spectrum is fast enough? 
        # Actually it calls MATLAB. That shouldn't be done in Main Thread if possible.
        # But if we are just plotting, maybe we wait for the next periodic update?
        pass

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
