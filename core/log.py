import logging
import sys
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class LogWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__(parent=None)
        self.setWindowTitle("Logger")
        self.resize(400, 300)
        self.main_window = main_window
        self.populate_window()

        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)

        self.handler = Handler(self)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def populate_window(self) -> None:
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create read-only text edit for log display
        self.log_ctrl = QTextEdit()
        self.log_ctrl.setReadOnly(True)
        layout.addWidget(self.log_ctrl)

    def clean(self):
        self.log_ctrl.clear()

    def log_message(self, message: str):
        try:
            self.log_ctrl.append(message)
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) or "deleted" in str(e):
                # Error comes up in jupyter notebooks because of stale GUI references and can be safely ignored
                pass
            else:
                raise

    def closeEvent(self, event):
        self.main_window.log_window = None
        event.accept()


class Handler(logging.StreamHandler):
    def __init__(self, log_window):
        super().__init__()
        self.log_window = log_window

    def emit(self, message) -> None:
        message = self.format(message)
        self.log_window.log_message(message)
