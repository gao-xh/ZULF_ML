import sys
from PySide6.QtWidgets import QApplication
from src.ui.optimization_window import OptimizationWindow

def main():
    app = QApplication(sys.argv)
    window = OptimizationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()