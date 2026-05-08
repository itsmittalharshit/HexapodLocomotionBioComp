from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import QSize
from OpenGL.GL import *
from core.world.world import World

class Canvas(QOpenGLWidget):
    def __init__(self, parent, size: QSize, world: World):
        super().__init__(parent)
        self.world: World = world
        self.initialised: bool = False
        # Don't set fixed size, allow resizing
        self.resize(size)

    def initializeGL(self) -> None:
        glViewport(0, 0, self.width(), self.height())
        glFinish()
        self.world._initialise_gl()
        glFinish()
        self.initialised = True

    def display(self) -> None:
        # Schedule a repaint which will call paintGL
        self.update()

    def paintGL(self) -> None:
        if self.initialised:
            self.world.display()

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, width, height)
        glFinish()

    def sizeHint(self) -> QSize:
        return QSize(800, 600)
