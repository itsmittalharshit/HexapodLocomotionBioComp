from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction, QMessageBox, QStatusBar
from PyQt5.QtCore import QPoint, QSize, QTimer, QThread, pyqtSignal, QMetaObject, Qt
import importlib.util
import sys
import time
import threading

from copy import deepcopy
from pathlib import Path
from gui.utils import AppIdentifiers as ID
from gui.canvas import Canvas
from core.world.world import World
from core.log import LogWindow

class Frame(QMainWindow):
    # Signal for thread-safe canvas updates
    update_canvas_signal = pyqtSignal()

    def __init__(
        self,
        title: str = "PyBEAST++",
        position: QPoint = QPoint(50, 50),
        size: QSize = QSize(808, 681),
        simulation = None
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.move(position)
        self.resize(size)

        self.simulation = simulation
        self.simulation_names = []
        self.simulation_class = []
        self.current_simulation = None
        self.world_canvas: Canvas = None
        self.current_thread: threading.Thread = None
        self.log_window = None
        self.current_simulation_id: int = -1
        self.fps = 60
        self.started, self.paused = False, False
        self.initial_simulation_copy = None

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect signal to slot
        self.update_canvas_signal.connect(self._update_canvas_slot)

        self.create_menu_bar()

    def load_demos(self) -> None:
        demo_directory = Path("./demos/")
        demos = [ f for f in demo_directory.iterdir() if f.suffix == ".py" ]
        for demo in demos:
            spec = importlib.util.spec_from_file_location(demo.stem, demo)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "IS_DEMO") and module.IS_DEMO:
                sys.modules[demo.stem] = module
                name = getattr(module, "DEMO_NAME") # Demo Title
                class_name = getattr(module, "CLASS_NAME") # Name of Simulation Class
                sim_class = getattr(module, class_name)
                self.simulation_names.append(name)
                self.simulation_class.append(sim_class)

    def create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        file_menu.addAction(about_action)

        # Demo menu
        self.load_demos()
        demo_menu = menu_bar.addMenu("&Demo")
        for i, demo in enumerate(self.simulation_names):
            demo_action = QAction(f"&{demo}", self)
            demo_action.setData(i)
            demo_action.triggered.connect(lambda checked, idx=i: self.on_start_demo(idx))
            demo_menu.addAction(demo_action)

        # Action menu
        action_menu = menu_bar.addMenu("&Action")
        if self.simulation is not None:
            start_action = QAction("&Start", self)
            start_action.triggered.connect(self.on_start_simulation)
            action_menu.addAction(start_action)

        pause_action = QAction("&Toggle Pause", self)
        pause_action.triggered.connect(self.on_pause)
        action_menu.addAction(pause_action)

        fast_action = QAction("&Toggle High Speed", self)
        fast_action.triggered.connect(self.on_fast)
        action_menu.addAction(fast_action)

        reset_action = QAction("&Reset", self)
        reset_action.triggered.connect(self.on_reset)
        action_menu.addAction(reset_action)

    def create_log_window(self) -> None:
        position = self.pos()
        size = self.size()
        self.log_window = LogWindow(self)
        self.log_window.move(position.x() + size.width() + 10, position.y())
        self.log_window.resize(400, size.height())
        self.log_window.show()

    def closeEvent(self, event):
        if self.current_thread is not None:
            self.kill_simulation()
        if self.log_window is not None:
            self.log_window.close()
        event.accept()

    def resizeEvent(self, event):
        if self.world_canvas is not None:
            if not self.paused:
                self.on_pause()
            # Canvas will handle its own resize through resizeGL
        super().resizeEvent(event)

    def on_start_simulation(self):
        self.start_simulation(self.simulation)

    def on_start_demo(self, index):
        simulation = self.simulation_class[index]()
        self.start_simulation(simulation)

    def on_pause(self) -> None:
        if self.current_simulation is None:
            return

        if not self.paused:
            self.pause_event.clear()
            self.paused = True
        else:
            self.pause_event.set()
            self.paused = False

    def on_fast(self) -> None:
        if self.render_simulation.is_set():
            self.render_simulation.clear()
        else:
            self.render_simulation.set()

    def on_about(self):
        QMessageBox.information(
            self,
            "PyBEAST++",
            """Bioinspired Evolutionary Agent Simulation Toolkit

PyBEAST++ Version: 1.0.0

PyBEAST developed by University of Leeds
PyBEAST++ created by James Borgars from PyBEAST"""
        )

    def on_reset(self) -> None:
        if self.initial_simulation_copy is not None:
            self.start_simulation(self.initial_simulation_copy)

    def start_simulation(self, simulation) -> None:
        self.initial_simulation_copy = deepcopy(simulation)

        if self.current_thread is not None:
            self.kill_simulation()

        self.current_simulation = simulation
        if not self.current_simulation.loaded:
            self.current_simulation.initialise()
        self.create_world_canvas(self.current_simulation.world)

        self.pause_event = threading.Event()
        self.render_simulation = threading.Event()
        self.kill_thread = threading.Event()
        self.current_thread = threading.Thread(
            target=self.run_simulation,
            args=(self.pause_event, self.render_simulation, self.kill_thread)
        )
        self.current_thread.daemon = True
        self.current_thread.start()

    def kill_simulation(self):
        self.kill_thread.set()
        self.current_thread.join()
        time.sleep(0.25)
        self.destroy_world_canvas()

    def _update_canvas_slot(self):
        """Slot to update canvas from the main thread"""
        if self.world_canvas is not None:
            self.world_canvas.display()

    def create_world_canvas(self, world: World):
        self.world_canvas = Canvas(self, self.size(), world)
        self.setCentralWidget(self.world_canvas)
        self.world_canvas.show()

    def destroy_world_canvas(self):
        if self.world_canvas is not None:
            self.world_canvas.close()
            self.world_canvas.deleteLater()
            self.world_canvas = None
        self.current_simulation = None
        self.current_thread = None

    def run_simulation(
        self,
        pause_event: threading.Event,
        render_simulation: threading.Event,
        kill_thread: threading.Event
    ) -> None:
        time.sleep(0.2)
        if not self.current_simulation.loaded:
            self.current_simulation.begin_simulation()
        else:
            self.current_simulation.resume_simulation()
        pause_event.set()
        render_simulation.set()

        complete = False
        while not complete:
            pause_event.wait()
            if kill_thread.is_set():
                break
            start_time = time.time()
            complete = self.current_simulation.update()
            if render_simulation.is_set():
                # Use signal for thread-safe GUI update
                self.update_canvas_signal.emit()
                sleep_for = max(0.01, (1.0 / self.fps) - (time.time() - start_time))
                time.sleep(sleep_for)
            if complete:
                break
