import sys
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QPushButton, QScrollArea, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)

def parse_isotopes(text):
    """Parse isotopes from comma-separated text"""
    return [s.strip() for s in text.replace('\n', ',').split(',') if s.strip()]

class VariableConfigDialog(QDialog):
    """Dialog to configure optimization parameters for variables."""
    def __init__(self, variables, parent=None):
        super().__init__(parent)
        self.variables = sorted(list(variables))
        self.config_data = {}
        self.setWindowTitle("Configure Variables")
        self.resize(800, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        layout.addWidget(QLabel("<b>Configure Optimization Parameters for Variables</b>"))
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Variable", "Initial Value", "Min", "Max", "Step Size", "Penalty Wt"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(len(self.variables))
        
        for i, var in enumerate(self.variables):
            self.table.setItem(i, 0, QTableWidgetItem(var))
            self.table.item(i, 0).setFlags(Qt.ItemIsEnabled) # Read only label

            # Defaults
            self.table.setItem(i, 1, QTableWidgetItem("10.0")) # Init
            self.table.setItem(i, 2, QTableWidgetItem("-500.0")) # Min
            self.table.setItem(i, 3, QTableWidgetItem("500.0")) # Max
            self.table.setItem(i, 4, QTableWidgetItem("0.5")) # Step
            self.table.setItem(i, 5, QTableWidgetItem("0.0")) # Elasticity
            
        layout.addWidget(self.table)
        
        # Buttons
        btns = QHBoxLayout()
        ok_btn = QPushButton("Confirm")
        ok_btn.clicked.connect(self.save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        layout.addLayout(btns)
        
    def save(self):
        try:
            config = {}
            for i in range(self.table.rowCount()):
                var = self.table.item(i, 0).text()
                init = float(self.table.item(i, 1).text())
                min_v = float(self.table.item(i, 2).text())
                max_v = float(self.table.item(i, 3).text())
                step = float(self.table.item(i, 4).text())
                elas = float(self.table.item(i, 5).text())
                
                config[var] = {
                    "initial_value": init,
                    "min_value": min_v,
                    "max_value": max_v,
                    "step_size": step,
                    "elasticity": elas
                }
            self.config_data = config
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Error", "All fields must be valid numbers.")

    def get_config(self):
        return self.config_data

class JCouplingEditorDialog(QDialog):
    """
    Popup window for editing the J-coupling matrix (Upper Triangle).
    Adapted from ZULF_NMR_Suite for standalone usage.
    """
    def __init__(self, isotopes, current_matrix=None, parent=None):
        super().__init__(parent)
        self.isotopes = isotopes
        
        # Determine number of spins
        self.n_spins = len(isotopes)
        
        # Initialize current values from matrix if provided
        self.current_values = {}
        if current_matrix is not None:
            current_matrix = np.array(current_matrix)
            if current_matrix.shape == (self.n_spins, self.n_spins):
                for i in range(self.n_spins):
                    for j in range(i + 1, self.n_spins):
                        self.current_values[(i, j)] = str(current_matrix[i, j])

        self.grid_inputs = {}  # Store input widgets {(i, j): QLineEdit}
        self.result_matrix = None # Will store the final numpy array upon Apply
        self.variable_config = None # Store variable config if present

        
        self.setWindowTitle("J-Coupling Editor")
        
         # Calculate optimal window size based on number of spins
        if self.n_spins <= 4:
            width, height = 800, 600
        elif self.n_spins <= 6:
            width, height = 900, 650
        else:
            width, height = 1100, 750
        self.resize(width, height)
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel(f"<h2 style='margin: 0; color: #1976D2;'>J-Coupling Matrix Editor</h2>")
        header.setStyleSheet("padding: 8px; background: #E3F2FD; border-radius: 6px;")
        layout.addWidget(header)
        
        # Info
        info_label = QLabel(
            f"<b>Editing upper triangle</b> for <b>{len(self.isotopes)} spins</b>: "
            f"<span style='color: #1976D2;'>{', '.join(self.isotopes)}</span><br>"
        )
        info_label.setStyleSheet("color: #546E7A; padding: 5px;")
        layout.addWidget(info_label)
        
        # Scroll Area for Grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(6)
        
        # --- Build Grid ---
        n = self.n_spins
        
        # Adjust size based on N
        font_size = 11 if n <= 6 else 9
        cell_min_w = 80
        
        # Header Row
        for j in range(n):
            lbl = QLabel(f"<b>{self.isotopes[j]}({j+1})</b>")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"background: #E3F2FD; border: 1px solid #90CAF9; font-size: {font_size}pt;")
            lbl.setMinimumWidth(cell_min_w)
            grid_layout.addWidget(lbl, 0, j + 1)
            
        # Rows
        for i in range(n):
            # Row Header
            lbl = QLabel(f"<b>{self.isotopes[i]}({i+1})</b>")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"background: #E3F2FD; border: 1px solid #90CAF9; font-size: {font_size}pt; padding-right: 5px;")
            lbl.setMinimumWidth(cell_min_w)
            grid_layout.addWidget(lbl, i + 1, 0)
            
            for j in range(n):
                if j > i: # Upper Triangle (Input)
                    inp = QLineEdit()
                    val = self.current_values.get((i, j), "0")
                    inp.setText(val)
                    inp.setAlignment(Qt.AlignCenter)
                    inp.setStyleSheet(f"border: 1px solid #CFD8DC; font-size: {font_size}pt; font-weight: bold;")
                    grid_layout.addWidget(inp, i + 1, j + 1)
                    self.grid_inputs[(i, j)] = inp
                    
                elif i == j: # Diagonal
                    lbl = QLabel("—")
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet("color: #BDBDBD; font-size: 14pt; background: #FAFAFA;")
                    grid_layout.addWidget(lbl, i + 1, j + 1)
                else: # Lower Triangle
                    lbl = QLabel("")
                    lbl.setStyleSheet("background: #F0F0F0;")
                    grid_layout.addWidget(lbl, i + 1, j + 1)
                    
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll)
        
        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 15px;")
        apply_btn.clicked.connect(self.apply_changes)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

    def apply_changes(self):
        """Construct symmetric matrix and accept."""
        # Detect variables vs numbers
        template = np.zeros((self.n_spins, self.n_spins), dtype=object)
        variables = set()
        
        try:
            for (i, j), inp in self.grid_inputs.items():
                txt = inp.text().strip()
                if not txt: txt = "0"
                
                try:
                    val = float(txt)
                    template[i, j] = val
                    template[j, i] = val
                except ValueError:
                    # It's a variable
                    variables.add(txt)
                    template[i, j] = txt
                    template[j, i] = txt # Symmetric
            
            if not variables:
                # Pure Numeric Mode (Legacy)
                self.result_matrix = template.astype(float)
                self.variable_config = None
                self.accept()
            else:
                # Variable Mode
                dlg = VariableConfigDialog(list(variables), self)
                if dlg.exec():
                    self.variable_config = dlg.get_config()
                    self.result_matrix = template # Keep as object array
                    self.accept()
                    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {e}")

