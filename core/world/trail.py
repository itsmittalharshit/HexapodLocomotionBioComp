from OpenGL.GL import *
from core.utils import Vec2

class Trail():
    def __init__(
        self,
        visible: bool = True,
        width: float = 2.0,
        length: int = 30,
        colour: list[float] = [1.0, 1.0, 1.0]
    ):
        self.colour = colour
        self.width = width
        self.length = length
        self.visible = visible
        self.points: list[Vec2] = []
        
    def display(self) -> None:
        if not self.visible or len(self.points) == 0:
            return
        glLineWidth(self.width)
        glEnable(GL_BLEND)
        glBegin(GL_LINE_STRIP)
        for i, pt in enumerate(self.points):
            glColor4f(self.colour[0], self.colour[1], self.colour[2], i / len(self.points))
            glVertex2d(pt[0], pt[1])
        glEnd()
        glDisable(GL_BLEND)
        glLineWidth(1.0)
    
    def append(self, location: Vec2) -> None:
        self.points.append(location)
    
    def update(self) -> None:
        while len(self.points) > self.length:
            self.points.pop(0)
    
    def clear(self):
        self.points.clear()
    
