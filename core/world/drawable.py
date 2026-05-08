import numpy as np

from abc import ABC
from OpenGL.GL import *
from core.utils import Vec2, DRAWABLE_RADIUS

class Drawable(ABC):
    def __init__(
        self,
        location: Vec2 = None,
        orientation: float = None,
        radius: float = DRAWABLE_RADIUS,
        visible: bool = True,
        colour: list[float] = [0.5, 0.5, 0.5, 1.0],
        edges: list[Vec2] = None
    ):
        self._start_location: Vec2 = location
        self._start_orientation: float = orientation
        self._display_list: int = 0

        self.location: Vec2 = location
        self.orientation: float = orientation
        self.radius: float = radius
        self.visible: bool = visible
        self.colour: list[float] = colour
        self.edges: list[Vec2] = edges
        
        self.circular = True if edges is None else False
        self.world = None
    
    def __del__(self):
        return
#        if self._display_list != 0:
#            glDeleteLists(self._display_list, 1)
    
    '''
    def _repr(self, **kwargs) -> str:
        class_name = self.__class__.__name__
        arg_string = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
        return f"{class_name}({arg_string})"
    
    def __repr__(self):
        return self._repr(
            location = self.location,
            orientation = self.orientation,
            radius = self.radius,
            colour = self.colour,
            edges = self.edges
        )
    '''
    
    def initialise(self) -> None:
#        if self._display_list != 0:
#            glDeleteLists(self._display_list, 1)
#        self._display_list = glGenLists(1)
#        glNewList(self._display_list, GL_COMPILE)
#        self.draw()
#       glEndList()
        
        if not self.circular:
            for e in self.edges:
                # If squared edge length > squared radius
                if np.linalg.norm(e)**2 > self.radius**2:
                    self.radius = np.linalg.norm(e)
    
    def display(self) -> None:
        if not self.visible or self.location is None:
            return
        glPushMatrix()
        glTranslated(self.location[0], self.location[1], 0.0)
        glRotated(np.rad2deg(self.orientation), 0.0, 0.0, 1.0)
        self.render()
        glPopMatrix()
    
    def render(self) -> None:
        self.draw()
#        if self._display_list:
#            glCallList(self._display_list)
    
    def draw(self) -> None:
        sides = 15 if self.circular else len(self.edges)
        glBegin(GL_POLYGON)
        for f in range(sides):
            pos: float = f / sides
            glColor4f(
                self.colour[0] * (1 - pos**2),
                self.colour[1] * (1 - pos**2),
                self.colour[2] * (1 - pos**2),
                self.colour[3]
            )
            if self.circular:
                glVertex2d(
                    self.radius * np.sin(pos * 2 * np.pi),
                    self.radius * np.cos(pos * 2 * np.pi)
                )
            else:
                glVertex2d(self.edges[f][0], self.edges[f][1])
                f += 1
        glEnd()
    
    def offset_orientation(self, angle: float) -> None:
        orientation = self.orientation + angle
        if orientation < 0 :
            orientation += 2 * np.pi
        orientation %= 2 * np.pi
        self.orientation = orientation

    
    # TODO: Serialiase/Deserialise?
