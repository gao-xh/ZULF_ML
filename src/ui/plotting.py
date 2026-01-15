import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
import matplotlib
# Ensure we use the QtAgg backend
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class SpectrumCanvas(FigureCanvas):
    """
    Custom Matplotlib Canvas for Spectrum Visualization.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        # Initialize parent
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initial Plot Styling
        self.ax.set_title("Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)

class SpectrumWidget(QWidget):
    """
    Complete Widget containing Canvas, Toolbar, and Coordinate Label.
    Designed to replace the raw Matplotlib code in OptimizationWindow.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Canvas
        self.canvas = SpectrumCanvas(self)
        
        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Coordinate Label
        self.coord_label = QLabel("Freq: - Hz | Amp: -")
        self.coord_label.setAlignment(Qt.AlignRight)
        self.coord_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        
        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)
        
        # Connect Events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # State storage (for redrawing/rescaling)
        self.current_data = {} 

    def on_mouse_move(self, event):
        """Update coordinate label on mouse hover."""
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.coord_label.setText(f"Freq: {x:.2f} Hz | Amp: {y:.2e}")
        else:
            self.coord_label.setText("Freq: - Hz | Amp: -")

    def update_plot(self, exp_freq, exp_amp, sim_freq=None, sim_amp=None, cost=None, iter_num=None):
        """
        Update the plot with new data.
        Retains zoom state if possible (Toolbar handles history, but re-plot might reset view).
        To keep view, we can get xlim/ylim before clearing.
        """
        ax = self.canvas.ax
        
        # Save current view limits if not autoscale
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        
        ax.clear()
        
        # Styles
        # Experimental: Black solid line, slightly transparent
        if exp_freq is not None:
             # Just filter for positive frequencies logic
             mask_exp = exp_freq >= 0
             e_f = exp_freq[mask_exp]
             e_a = exp_amp[mask_exp]
             ax.plot(e_f, e_a, 'k-', linewidth=1.2, alpha=0.6, label='Experiment')
        
        # Simulated: Red dashed line
        if sim_freq is not None and sim_amp is not None:
             # Just filter for positive frequencies logic
             mask_sim = sim_freq >= 0
             s_f = sim_freq[mask_sim]
             s_a = sim_amp[mask_sim]

             # Auto-scale simulation to match experiment max (Visualization only)
             scale = 1.0
             # Note: exp_amp might be None or full array, better use e_a if defined
             max_exp_amp = np.max(e_a) if 'e_a' in locals() and len(e_a) > 0 else (np.max(exp_amp) if exp_amp is not None else 0)
             
             if np.max(s_a) > 0 and max_exp_amp > 0:
                 scale = max_exp_amp / np.max(s_a)
             
             ax.plot(s_f, s_a * scale, 'r--', linewidth=1.0, label=f'Simulated (x{scale:.2f})')
             
        # Labels and Title
        title = "Optimization Progress"
        if iter_num is not None and cost is not None:
            title = f"Iter {iter_num} | Cost: {cost:.4f}"
        
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Force Positive Frequencies only (as requested)
        ax.set_xlim(left=0)
        
        # Restore view if needed, or let autoscale handle it?
        # Usually during optimization, autoscale is better to see the peaks forming.
        # But if user zooms in, we shouldn't reset it every step.
        # However, NavigationToolbar has "Home" to reset.
        # Let's trust matplotlib's behavior: .plot() usually resets limits.
        
        # To preserve zoom:
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        # BUT: this prevents auto-scaling if peaks move or grow.
        # We can detect if the user has zoomed (toolbar history).
        # For now, let's stick to auto-scale for "Real-time" monitoring. 
        # User allows to pause and zoom.
        
        self.canvas.draw()
        
    def clear(self):
        self.canvas.ax.clear()
        self.canvas.ax.grid(True)
        self.canvas.draw()
