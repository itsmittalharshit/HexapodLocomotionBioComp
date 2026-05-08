import sys
from PyQt5.QtWidgets import QApplication

from gui.frame import Frame

class App(QApplication):
    def __init__(self, simulation=None):
        super().__init__(sys.argv)
        self.simulation = simulation
        self.frame = Frame(simulation=self.simulation)
        self.frame.show()


if __name__ == "__main__":
    app = App()
    sys.exit(app.exec_())
