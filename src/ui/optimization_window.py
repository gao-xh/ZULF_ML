import csv
import json
import sys

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from src.config import OptimizationStageConfig, OptimizerConfig
from src.core.optimizer import ZulfOptimizer
from src.core.simulation_wrapper import available_backends
from src.ui.molecule_editor import JCouplingEditorDialog, parse_isotopes
from src.ui.plotting import SpectrumWidget
from src.utils.loaders import load_experimental_and_config

# ---------- Worker Thread ----------
class OptimizationWorker(QThread):
    """
    Worker thread that runs the ZulfOptimizer loop.
    Emits signals for progress updates and completion.
    """
    log = Signal(str)
    progress = Signal(int, float, object)            # iteration, current_cost, status_data
    new_best = Signal(int, float, object, object)    # iteration, best_cost, best_params, viz_data
    finished = Signal(object, list)          # best_params, history
    failed = Signal(str)

    def __init__(self, optimizer, init_params, freq_range=None, variable_config=None, resume_state=None):
        super().__init__()
        self.optimizer = optimizer
        self.init_params = init_params
        self.freq_range = freq_range
        self.variable_config = variable_config
        self.resume_state = resume_state
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
                variable_config=self.variable_config,
                resume_state=self.resume_state,
            )
            self.finished.emit(best_params, history)
        except Exception as e:
            self.failed.emit(str(e))

    def stop(self):
        self._is_running = False

    def _step_callback(self, i, curr_cost, best_cost, best_params, viz_data=None, status_data=None):
        if not self._is_running:
            return False # Stop optimizer
        
        self.progress.emit(i, curr_cost, status_data or {})
        
        # Emit new best if applicable (cost has improved)
        if best_cost == curr_cost and viz_data is not None: 
             self.new_best.emit(i, best_cost, best_params, viz_data)
             
        return True


class OptimizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML_ZULF Staged Optimizer")
        self.resize(1380, 860)

        self.optimizer = None
        self.worker = None
        self.backend_names = available_backends()
        self.stage_presets = {
            'Two-stage default': {
                'a': ('fast_eigen', 0.4, (0.85, 0.15, 0.0)),
                'b': ('python_fid', 0.6, (0.2, 0.65, 0.15)),
                'c': ('spinach', 0.0, (0.15, 0.7, 0.15)),
            },
            'Three-stage validation': {
                'a': ('fast_eigen', 0.3, (0.85, 0.15, 0.0)),
                'b': ('python_fid', 0.5, (0.2, 0.65, 0.15)),
                'c': ('spinach', 0.2, (0.15, 0.7, 0.15)),
            },
            'Python-only refinement': {
                'a': ('fast_eigen', 0.35, (0.85, 0.15, 0.0)),
                'b': ('python_fid', 0.65, (0.2, 0.65, 0.15)),
                'c': ('python_fid', 0.0, (0.15, 0.7, 0.15)),
            },
        }

        self.exp_spectrum = None
        self.exp_fid = None
        self.sampling_rate = None
        self.j_coupling = None
        self.variable_config = None
        self.best_viz_data = None
        self.best_params_result = None
        self.best_cost_result = None
        self.current_stage_name = None
        self.pending_resume_state = None
        self.latest_cost_components = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(470)
        left_container = QWidget()
        self.left_layout = QVBoxLayout(left_container)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)
        left_scroll.setWidget(left_container)
        main_layout.addWidget(left_scroll, 0)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        main_layout.addWidget(plot_panel, 1)

        self._build_data_group()
        self._build_run_group()
        self._build_stage_group()
        self._build_cost_group()
        self._build_status_group()
        self._build_control_group()
        self._build_log_group()
        self.left_layout.addStretch(1)

        self.plot_widget = SpectrumWidget(self)
        plot_layout.addWidget(self.plot_widget)

        self.btn_load_exp.clicked.connect(self.load_experiment)
        self.btn_load_mol.clicked.connect(self.load_molecule)
        self.btn_top_sys.clicked.connect(self.open_system_builder)
        self.btn_start.clicked.connect(self.start_optimization)
        self.btn_stop.clicked.connect(self.stop_optimization)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save_checkpoint.clicked.connect(self.save_checkpoint)
        self.btn_load_checkpoint.clicked.connect(self.load_checkpoint)

        self.update_stage_summary()
        self.update_status_panel()

    def _make_int_spin(self, minimum, maximum, value, step=1):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def _make_float_spin(self, minimum, maximum, value, step=0.1, decimals=3):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def _make_backend_combo(self, default_name):
        combo = QComboBox()
        combo.addItems(self.backend_names)
        combo.setCurrentText(default_name if default_name in self.backend_names else self.backend_names[0])
        combo.currentTextChanged.connect(self.update_stage_summary)
        return combo

    def _build_data_group(self):
        data_group = QGroupBox("Data")
        form = QFormLayout(data_group)

        self.btn_load_exp = QPushButton("Load Experiment Folder")
        self.lbl_exp_status = QLabel("No experimental dataset loaded")
        self.lbl_exp_status.setWordWrap(True)

        self.btn_load_mol = QPushButton("Load Molecule CSV")
        self.lbl_mol_status = QLabel("No molecule loaded")
        self.lbl_mol_status.setWordWrap(True)

        self.btn_top_sys = QPushButton("Build Molecule Manually")
        self.lbl_system_mode = QLabel("Mode: not configured")

        form.addRow(self.btn_load_exp, self.lbl_exp_status)
        form.addRow(self.btn_load_mol, self.lbl_mol_status)
        form.addRow(self.btn_top_sys, self.lbl_system_mode)
        self.left_layout.addWidget(data_group)

    def _build_run_group(self):
        run_group = QGroupBox("Run Settings")
        form = QFormLayout(run_group)

        self.spin_steps = self._make_int_spin(1, 10000, 120, 10)
        self.spin_plot_interval = self._make_int_spin(1, 1000, 10, 1)
        self.spin_t2 = self._make_float_spin(0.01, 20.0, 0.8, 0.1, 3)
        self.spin_sg = self._make_int_spin(5, 501, 101, 2)
        self.spin_trunc = self._make_int_spin(0, 5000, 160, 10)
        self.spin_freq_min = self._make_float_spin(0.0, 10000.0, 0.0, 5.0, 1)
        self.spin_freq_max = self._make_float_spin(0.0, 10000.0, 400.0, 5.0, 1)

        form.addRow("Iterations", self.spin_steps)
        form.addRow("Plot Interval", self.spin_plot_interval)
        form.addRow("Initial T2 / linewidth", self.spin_t2)
        form.addRow("Initial SG Window", self.spin_sg)
        form.addRow("Initial Truncation", self.spin_trunc)
        form.addRow("Fit Freq Min", self.spin_freq_min)
        form.addRow("Fit Freq Max", self.spin_freq_max)
        self.left_layout.addWidget(run_group)

    def _build_cost_group(self):
        cost_group = QGroupBox("Cost Tuning")
        form = QFormLayout(cost_group)

        self.spin_missing_peak_penalty = self._make_float_spin(0.0, 20.0, 1.0, 0.1, 2)
        self.spin_peak_region_weight = self._make_float_spin(0.0, 20.0, 3.0, 0.1, 2)
        self.lbl_cost_components = QLabel("c1: -, c2: -, c3: -")
        self.lbl_cost_components.setWordWrap(True)

        form.addRow("Missing Peak Penalty", self.spin_missing_peak_penalty)
        form.addRow("Peak Region Weight", self.spin_peak_region_weight)
        form.addRow("Live Components", self.lbl_cost_components)
        self.left_layout.addWidget(cost_group)

    def _build_stage_group(self):
        stage_group = QGroupBox("Staged Optimization")
        stage_layout = QVBoxLayout(stage_group)

        preset_row = QHBoxLayout()
        self.combo_stage_preset = QComboBox()
        self.combo_stage_preset.addItems(self.stage_presets.keys())
        self.btn_apply_stage_preset = QPushButton("Apply Preset")
        self.btn_apply_stage_preset.clicked.connect(self.apply_stage_preset)
        self.btn_normalize_stages = QPushButton("Normalize Fractions")
        self.btn_normalize_stages.clicked.connect(self.normalize_stage_fractions)
        preset_row.addWidget(QLabel("Preset"))
        preset_row.addWidget(self.combo_stage_preset, 1)
        preset_row.addWidget(self.btn_apply_stage_preset)
        preset_row.addWidget(self.btn_normalize_stages)
        stage_layout.addLayout(preset_row)

        stage_layout.addWidget(self._build_stage_form(
            title="Stage A: Coarse Search",
            backend_default='fast_eigen',
            fraction_default=0.4,
            weights_default=(1.0, 0.0, 0.0),
            prefix='a',
        ))
        stage_layout.addWidget(self._build_stage_form(
            title="Stage B: Refinement",
            backend_default='python_fid',
            fraction_default=0.6,
            weights_default=(0.6, 0.3, 0.1),
            prefix='b',
        ))
        stage_layout.addWidget(self._build_stage_form(
            title="Stage C: Validation / Final Ranking",
            backend_default='spinach',
            fraction_default=0.0,
            weights_default=(0.6, 0.3, 0.1),
            prefix='c',
        ))

        self.lbl_stage_summary = QLabel()
        self.lbl_stage_summary.setWordWrap(True)
        stage_layout.addWidget(self.lbl_stage_summary)
        self.left_layout.addWidget(stage_group)

    def _build_stage_form(self, title, backend_default, fraction_default, weights_default, prefix):
        box = QGroupBox(title)
        form = QFormLayout(box)

        backend_combo = self._make_backend_combo(backend_default)
        fraction_spin = self._make_float_spin(0.0, 1.0, fraction_default, 0.05, 2)
        weight_pos = self._make_float_spin(0.0, 5.0, weights_default[0], 0.1, 2)
        weight_l2 = self._make_float_spin(0.0, 5.0, weights_default[1], 0.1, 2)
        weight_height = self._make_float_spin(0.0, 5.0, weights_default[2], 0.1, 2)

        fraction_spin.valueChanged.connect(self.update_stage_summary)
        weight_pos.valueChanged.connect(self.update_stage_summary)
        weight_l2.valueChanged.connect(self.update_stage_summary)
        weight_height.valueChanged.connect(self.update_stage_summary)

        setattr(self, f'combo_stage_{prefix}_backend', backend_combo)
        setattr(self, f'spin_stage_{prefix}_fraction', fraction_spin)
        setattr(self, f'spin_stage_{prefix}_pos', weight_pos)
        setattr(self, f'spin_stage_{prefix}_l2', weight_l2)
        setattr(self, f'spin_stage_{prefix}_height', weight_height)

        form.addRow("Backend", backend_combo)
        form.addRow("Fraction", fraction_spin)
        form.addRow("Weight Pos", weight_pos)
        form.addRow("Weight L2", weight_l2)
        form.addRow("Weight Height", weight_height)
        return box

    def _build_status_group(self):
        status_group = QGroupBox("Session Status")
        form = QFormLayout(status_group)

        self.lbl_backend_status = QLabel("No optimizer running")
        self.lbl_backend_status.setWordWrap(True)
        self.lbl_stage_status = QLabel("Stage: idle")
        self.lbl_stage_status.setWordWrap(True)
        self.lbl_result_status = QLabel("Best result: none")
        self.lbl_result_status.setWordWrap(True)

        form.addRow("Backend Plan", self.lbl_backend_status)
        form.addRow("Current Stage", self.lbl_stage_status)
        form.addRow("Best Result", self.lbl_result_status)
        self.left_layout.addWidget(status_group)

    def _build_control_group(self):
        control_group = QGroupBox("Actions")
        layout = QHBoxLayout(control_group)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("Save Report")
        self.btn_save.setEnabled(False)
        self.btn_save_checkpoint = QPushButton("Save Checkpoint")
        self.btn_save_checkpoint.setEnabled(False)
        self.btn_load_checkpoint = QPushButton("Load Checkpoint")

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_save_checkpoint)
        layout.addWidget(self.btn_load_checkpoint)
        self.left_layout.addWidget(control_group)

    def apply_optimizer_config(self, config):
        self.spin_steps.setValue(config.max_iterations)
        self.spin_plot_interval.setValue(config.plot_interval)
        self.spin_missing_peak_penalty.setValue(config.cost_function.missing_peak_penalty)
        self.spin_peak_region_weight.setValue(config.cost_function.peak_region_weight)

        stage_defaults = {
            'a': OptimizationStageConfig('stage_a', self.backend_names[0], 0.0, (0.0, 0.0, 0.0)),
            'b': OptimizationStageConfig('stage_b', self.backend_names[0], 0.0, (0.0, 0.0, 0.0)),
            'c': OptimizationStageConfig('stage_c', self.backend_names[0], 0.0, (0.0, 0.0, 0.0)),
        }
        for prefix, stage in zip(('a', 'b', 'c'), config.stages):
            stage_defaults[prefix] = stage

        for prefix, stage in stage_defaults.items():
            self._set_stage_form(prefix, stage.backend_name, stage.fraction, stage.weights)

        self.update_stage_summary()

    def _build_log_group(self):
        log_group = QGroupBox("Log")
        layout = QVBoxLayout(log_group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        layout.addWidget(self.text_log)
        self.left_layout.addWidget(log_group)

    def build_stage_configs(self):
        stage_configs = []
        for prefix, name in (('a', 'stage_a'), ('b', 'stage_b'), ('c', 'stage_c')):
            fraction = getattr(self, f'spin_stage_{prefix}_fraction').value()
            stage_configs.append(OptimizationStageConfig(
                name=name,
                backend_name=getattr(self, f'combo_stage_{prefix}_backend').currentText(),
                fraction=fraction,
                weights=(
                    getattr(self, f'spin_stage_{prefix}_pos').value(),
                    getattr(self, f'spin_stage_{prefix}_l2').value(),
                    getattr(self, f'spin_stage_{prefix}_height').value(),
                ),
            ))
        return stage_configs

    def normalized_stage_configs(self):
        stages = self.build_stage_configs()
        active_total = sum(stage.fraction for stage in stages if stage.fraction > 0)
        if active_total <= 0:
            return stages

        normalized = []
        for stage in stages:
            fraction = stage.fraction / active_total if stage.fraction > 0 else 0.0
            normalized.append(OptimizationStageConfig(
                name=stage.name,
                backend_name=stage.backend_name,
                fraction=fraction,
                weights=stage.weights,
            ))
        return normalized

    def _set_stage_form(self, prefix, backend_name, fraction, weights):
        getattr(self, f'combo_stage_{prefix}_backend').setCurrentText(backend_name)
        getattr(self, f'spin_stage_{prefix}_fraction').setValue(fraction)
        getattr(self, f'spin_stage_{prefix}_pos').setValue(weights[0])
        getattr(self, f'spin_stage_{prefix}_l2').setValue(weights[1])
        getattr(self, f'spin_stage_{prefix}_height').setValue(weights[2])

    def apply_stage_preset(self):
        preset_name = self.combo_stage_preset.currentText()
        preset = self.stage_presets.get(preset_name)
        if not preset:
            return

        for prefix in ('a', 'b', 'c'):
            backend_name, fraction, weights = preset[prefix]
            self._set_stage_form(prefix, backend_name, fraction, weights)

        self.log(f"Applied stage preset: {preset_name}")
        self.update_stage_summary()

    def normalize_stage_fractions(self):
        normalized = self.normalized_stage_configs()
        active = [(prefix, stage) for prefix, stage in zip(('a', 'b', 'c'), normalized) if stage.fraction > 0]
        assigned = 0.0

        for index, (prefix, stage) in enumerate(zip(('a', 'b', 'c'), normalized)):
            spin = getattr(self, f'spin_stage_{prefix}_fraction')
            if stage.fraction <= 0:
                spin.setValue(0.0)
                continue

            active_index = next(i for i, (active_prefix, _) in enumerate(active) if active_prefix == prefix)
            if active_index == len(active) - 1:
                value = max(0.0, round(1.0 - assigned, 2))
            else:
                value = round(stage.fraction, 2)
                assigned += value

            spin.setValue(value)

        self.log("Normalized active stage fractions to sum to 1.00")
        self.update_stage_summary()

    def update_stage_summary(self):
        stages = self.build_stage_configs()
        normalized = self.normalized_stage_configs()
        total_fraction = sum(stage.fraction for stage in stages)
        summary_lines = []
        for stage, normalized_stage in zip(stages, normalized):
            state = 'active' if stage.fraction > 0 else 'inactive'
            summary_lines.append(
                f"{stage.name} [{state}]: {stage.backend_name}, fraction={stage.fraction:.2f}, normalized={normalized_stage.fraction:.2f}, weights={stage.weights}"
            )
        summary_lines.append(f"Total fraction: {total_fraction:.2f}")
        self.lbl_stage_summary.setText("\n".join(summary_lines))
        self.update_status_panel()

    def update_status_panel(self):
        stages = self.normalized_stage_configs()
        active_stages = [stage for stage in stages if stage.fraction > 0]
        inactive_stages = [stage.name for stage in stages if stage.fraction <= 0]
        backend_summary = " -> ".join(f"{stage.name}:{stage.backend_name}" for stage in active_stages)
        if inactive_stages:
            inactive_text = ", ".join(inactive_stages)
            backend_summary = f"{backend_summary} | inactive: {inactive_text}" if backend_summary else f"inactive: {inactive_text}"
        self.lbl_backend_status.setText(backend_summary if backend_summary else "No stages configured")
        current_stage = self.current_stage_name or "idle"
        self.lbl_stage_status.setText(current_stage)
        if self.best_cost_result is None:
            self.lbl_result_status.setText("Best result: none")
        else:
            self.lbl_result_status.setText(f"Best cost: {self.best_cost_result:.4f}")

    def update_cost_component_panel(self, components):
        self.latest_cost_components = components
        if not components:
            self.lbl_cost_components.setText("c1: -, c2: -, c3: -")
            return
        self.lbl_cost_components.setText(
            f"c1={components.get('c1', float('nan')):.4f} | "
            f"c2={components.get('c2', float('nan')):.4f} | "
            f"c3={components.get('c3', float('nan')):.4f}"
        )

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
                self.pending_resume_state = None
                
                sp_len = len(spectrum[0]) if spectrum else 0
                fid_len = len(fid) if fid is not None else 0
                
                self.lbl_exp_status.setText(f"Spectrum: {sp_len} pts | FID: {fid_len} pts")
                self.log(f"Loaded data from {folder}")
                self.log(f"Sampling Rate: {sr} Hz")
                self.update_status_panel()
                
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
                 self.variable_config = None
                 self.pending_resume_state = None
                 
                 self.lbl_mol_status.setText(f"Loaded: {len(self.isotopes)} spins")
                 self.lbl_system_mode.setText("Mode: Numeric")
                 self.log(f"Loaded molecule from {path}. Mode: Numeric")
                 self.update_status_panel()
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
            
        current_j = None
        if self.j_coupling is not None and self.j_coupling.shape[0] == len(isotopes):
            current_j = self.j_coupling
            
        dlg = JCouplingEditorDialog(isotopes, current_j, self)
        if dlg.exec():
            # 3. Retrieve Result
            self.isotopes = isotopes
            self.j_coupling = dlg.result_matrix
            self.variable_config = dlg.variable_config
            self.pending_resume_state = None
            
            n_spins = len(self.isotopes)
            mode = "Variable Mode" if self.variable_config else "Numeric Mode"
            self.lbl_mol_status.setText(f"Manual: {n_spins} spins ({mode})")
            self.lbl_system_mode.setText(f"Mode: {mode}")
            self.log(f"System defined manually: {isotopes}")
            self.log(f"J-Coupling Matrix updated ({n_spins}x{n_spins}). Mode: {mode}")
            self.update_status_panel()

    def start_optimization(self):
        if (self.exp_spectrum is None and self.exp_fid is None) or self.j_coupling is None:
            self.log("Error: Please load both Experiment Data and Molecule Structure.")
            return

        try:
            config = OptimizerConfig()
            config.max_iterations = self.spin_steps.value()
            config.plot_interval = self.spin_plot_interval.value()
            config.stages = self.normalized_stage_configs()

            total_fraction = sum(stage.fraction for stage in config.stages)
            if total_fraction <= 0:
                self.log("Error: stage fractions must sum to a positive value.")
                return

            if abs(total_fraction - 1.0) > 1e-6:
                self.log(f"Normalized stage fractions internally from total {total_fraction:.2f}")

            config.t2_linewidth.initial_value = self.spin_t2.value()
            config.sg_window.initial_value = self.spin_sg.value()
            config.truncation.initial_value = self.spin_trunc.value()
            config.cost_function.missing_peak_penalty = self.spin_missing_peak_penalty.value()
            config.cost_function.peak_region_weight = self.spin_peak_region_weight.value()
            
            spins = getattr(self, 'isotopes', ['1H'] * self.j_coupling.shape[0])
            sr = self.sampling_rate if self.sampling_rate else 400.0

            try:
                self.optimizer = ZulfOptimizer(
                    spins=spins,
                    sampling_rate=sr,
                    exp_spectrum=self.exp_spectrum,
                    exp_fid=self.exp_fid,
                    backend_name=config.stages[0].backend_name,
                )
            except Exception as e:
                self.log(f"Optimizer instantiation failed: {e}")
                return
            
            self.optimizer.config = config
            resume_state = self.pending_resume_state

            if resume_state is not None:
                init_params = resume_state.get('initial_params') or resume_state['current_params']
            else:
                init_params = (
                    self.j_coupling,
                    config.sg_window.initial_value,
                    config.truncation.initial_value,
                    config.t2_linewidth.initial_value
                )
            
            f_min = self.spin_freq_min.value()
            f_max = self.spin_freq_max.value()
            freq_range = (f_min, f_max)

            var_config = getattr(self, 'variable_config', None)
            
            self.worker = OptimizationWorker(self.optimizer, init_params, freq_range, var_config, resume_state)
            self.worker.log.connect(self.log)
            self.worker.progress.connect(self.on_progress)
            self.worker.new_best.connect(self.on_new_best)
            self.worker.finished.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            
            self.worker.start()
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save_checkpoint.setEnabled(False)
            self.current_stage_name = resume_state.get('current_stage_name') if resume_state is not None else config.stages[0].name
            self.best_cost_result = None
            self.update_cost_component_panel(None)
            self.update_status_panel()
            if resume_state is not None:
                self.log(f"Resuming optimization from iteration {resume_state['completed_iterations']}")
            else:
                self.log("Optimization started...")
            for stage in config.stages:
                self.log(
                    f"Configured {stage.name}: backend={stage.backend_name}, fraction={stage.fraction:.2f}, weights={stage.weights}"
                )
            
        except Exception as e:
            self.log(f"Failed to start: {e}")

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stopping...")

    def on_progress(self, iteration, cost, status_data):
        stage_name = status_data.get('stage') if status_data else None
        if stage_name:
            self.current_stage_name = stage_name
        self.update_cost_component_panel(status_data.get('components') if status_data else None)
        if iteration % 10 == 0:
            stage_name = self.current_stage_name or "unknown"
            self.log(f"Iter {iteration}: Cost {cost:.4f} | Stage {stage_name}")

    def on_new_best(self, iteration, cost, params, viz_data):
        stage_name = viz_data.get('stage') if viz_data else None
        if stage_name:
            self.current_stage_name = stage_name
        self.log(f"New Best found at iter {iteration} (Cost: {cost:.4f})")

        self.best_viz_data = viz_data
        self.best_params_result = params
        self.best_cost_result = cost
        self.update_cost_component_panel(viz_data.get('components') if viz_data else None)
        self.update_status_panel()

        if viz_data:
            try:
                sim_freq = viz_data['sim_freq']
                sim_amp = viz_data['sim_amp']
                exp_freq = viz_data['exp_freq']
                exp_amp = viz_data['exp_amp']
                stage_name = viz_data.get('stage')
                
                self.plot_widget.update_plot(
                    exp_freq, exp_amp, 
                    sim_freq, sim_amp, 
                    cost=cost, iter_num=iteration
                )
                if stage_name:
                    self.log(f"Visualization updated for {stage_name}")
                
            except Exception as e:
                print(f"Plotting error: {e}")

    def on_finished(self, best_params, history):
        self.log("Optimization Finished.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.btn_save_checkpoint.setEnabled(self.optimizer is not None and self.optimizer.current_params is not None)
        self.worker = None
        self.pending_resume_state = None
        if self.optimizer and self.optimizer.stage_history:
            self.current_stage_name = self.optimizer.stage_history[-1]
        self.update_status_panel()

    def on_failed(self, error):
        self.log(f"Optimization Failed: {error}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.best_params_result is not None:
             self.btn_save.setEnabled(True)
        self.btn_save_checkpoint.setEnabled(self.optimizer is not None and self.optimizer.current_params is not None)
        self.worker = None
        self.update_status_panel()

    def save_checkpoint(self):
        if self.worker and self.worker.isRunning():
            self.log("Checkpoint save is only enabled after the current run stops.")
            return
        if self.optimizer is None or self.optimizer.current_params is None:
            self.log("No optimizer state available for checkpointing.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "", "Checkpoint Files (*.npz)"
        )
        if not path:
            return
        if not path.lower().endswith('.npz'):
            path = f"{path}.npz"

        try:
            self.optimizer.save_checkpoint(path)
            self.log(f"Checkpoint saved to {path}")
        except Exception as e:
            self.log(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        if self.worker and self.worker.isRunning():
            self.log("Stop the current run before loading a checkpoint.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", "", "Checkpoint Files (*.npz)"
        )
        if not path:
            return

        try:
            state = ZulfOptimizer.load_checkpoint(path)
            self.pending_resume_state = state
            self.exp_fid = state['exp_fid']
            self.exp_spectrum = state['exp_spectrum']
            self.sampling_rate = state['sampling_rate']
            self.isotopes = state['spins']
            self.variable_config = None

            initial_params = state.get('initial_params')
            if initial_params is not None:
                self.j_coupling = np.array(initial_params[0])
            else:
                self.j_coupling = np.array(state['current_params'][0])

            config = state['config']
            self.apply_optimizer_config(config)

            self.spin_t2.setValue(state['current_params'][3])
            self.spin_sg.setValue(state['current_params'][1])
            self.spin_trunc.setValue(state['current_params'][2])

            freq_range = state.get('freq_range')
            if freq_range:
                if freq_range[0] is not None:
                    self.spin_freq_min.setValue(freq_range[0])
                if freq_range[1] is not None:
                    self.spin_freq_max.setValue(freq_range[1])

            spectrum_points = len(self.exp_spectrum[0]) if self.exp_spectrum is not None else 0
            fid_points = len(self.exp_fid) if self.exp_fid is not None else 0
            self.lbl_exp_status.setText(
                f"Checkpoint dataset | Spectrum: {spectrum_points} pts | FID: {fid_points} pts"
            )
            self.lbl_mol_status.setText(f"Checkpoint: {len(self.isotopes)} spins")
            self.lbl_system_mode.setText("Mode: Numeric")
            self.best_cost_result = state['best_cost']
            self.best_params_result = state['best_params']
            self.current_stage_name = state.get('current_stage_name') or 'resume-ready'
            self.update_cost_component_panel(None)
            if self.exp_spectrum is not None:
                self.plot_widget.update_plot(self.exp_spectrum[0], self.exp_spectrum[1])
            self.btn_save.setEnabled(False)
            self.btn_save_checkpoint.setEnabled(False)
            self.update_status_panel()
            self.log(f"Loaded checkpoint from {path}")
            self.log(f"Resume is ready from iteration {state['completed_iterations']}")
        except Exception as e:
            self.log(f"Error loading checkpoint: {e}")

    def save_results(self):
        if not self.best_viz_data or not self.best_params_result:
            self.log("No results to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Optimization Report", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if not path:
            return

        try:
            final_j, final_sg, final_trunc, final_t2 = self.best_params_result
            sim_freq = self.best_viz_data['sim_freq']
            sim_amp = self.best_viz_data['sim_amp']
            exp_freq = self.best_viz_data['exp_freq']
            exp_amp = self.best_viz_data['exp_amp']
            stage_name = self.best_viz_data.get('stage', self.current_stage_name)
            scale = 1.0
            if np.max(sim_amp) > 0 and np.max(exp_amp) > 0:
                scale = np.max(exp_amp) / np.max(sim_amp)
            
            if path.lower().endswith('.csv'):
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["# Optimization Report"])
                    writer.writerow(["# Final Cost", self.best_cost_result])
                    writer.writerow(["# Stage", stage_name])
                    writer.writerow(["# T2 Linewidth (Hz)", final_t2])
                    writer.writerow(["# SG Window", final_sg])
                    writer.writerow(["# Truncation", final_trunc])
                    writer.writerow([])
                    
                    writer.writerow(["# J-Coupling Matrix"])
                    for row in final_j:
                        writer.writerow(["# J_Row"] + list(row))
                    writer.writerow([])
                    
                    writer.writerow(["Frequency (Hz)", "Experimental Amp", "Simulated Amp"])
                    min_len = min(len(exp_freq), len(sim_freq))

                    for i in range(min_len):
                        writer.writerow([exp_freq[i], exp_amp[i], sim_amp[i] * scale])
                        
                self.log(f"Saved CSV report to {path}")

            elif path.lower().endswith('.json'):
                report = {
                    "final_cost": float(self.best_cost_result),
                    "stage": stage_name,
                    "parameters": {
                        "t2_linewidth": float(final_t2),
                        "sg_window": int(final_sg),
                        "truncation": int(final_trunc),
                        "j_coupling": final_j.tolist()
                    },
                    "spectra": {
                         "frequency": exp_freq.tolist(),
                         "experimental": exp_amp.tolist(),
                        "simulated": (sim_amp * scale).tolist()
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
