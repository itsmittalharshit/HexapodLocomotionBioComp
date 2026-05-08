import numpy as np

from OpenGL.GL import *
from core.sensor.base import Sensor, MatchFunction, EvaluateFunction, ScaleFunction
from core.utils import Vec2, BeamSettings as BS, get_vector_angle
from numpy import rad2deg

class BeamSensor(Sensor):
    def __init__(
        self,
        scope: float = np.pi / 4,
        distance: float = 250.0,
        location: Vec2 = None,
        orientation: float = 0.0,
        relative_location: Vec2 = None,
        relative_orientation: float = 0,
        match_function: MatchFunction = None,
        evaluate_function: EvaluateFunction = None,
        scale_function: ScaleFunction = None,
        wrap: bool = False,
        draw_fixed: bool = True,
        draw_scale: float = 1.0,
        beam_quality: float = BS.BEAM_QUALITY
    ):
        super().__init__(
            location,
            orientation,
            relative_location,
            relative_orientation,
            match_function,
            evaluate_function,
            scale_function
        )
        assert 0.0 <= scope <= 2*np.pi
        self.scope = scope
        self.range = distance
        self.draw_scale = draw_scale
        self.draw_fixed = draw_fixed
        self.wrap = wrap
        self.wrapping = {
            "Left": False,
            "Right": False,
            "Bottom": False,
            "Top": False
        }
        self._beam_quality = beam_quality
    
    def __repr__(self):
        return self._repr(
            scope = self.scope,
            range = self.range,
            owner = self.owner
        )
    
    def update(self) -> None:
        if self.wrap:
            self.wrapping["Left"] = self.location[0] - self.range < 0
            self.wrapping["Bottom"] = self.location[1] - self.range < 0
            self.wrapping["Right"] = self.location[0] + self.range > self.owner.world.width
            self.wrapping["Top"] = self.location[1] + self.range > self.owner.world.height
        super().update()
    
    def _display(self) -> None:
        glPushMatrix()
        glTranslated(self.location[0], self.location[1], 0)
        glRotated(rad2deg(self.orientation), 0.0, 0.0, 1.0)
        
        if self.draw_fixed:
            scale = self.draw_scale
        else:
            scale = self.draw_scale - self.output()
        
        glScaled(scale, scale, 1.0)
        self.draw()
#        glCallList(self._display_list)
        glPopMatrix()
        
    def display(self) -> None:
        self._display()
        if not self.wrap:
            return
        if self.wrapping["Left"]:
            temp = self.location[0]
            self.location[0] = temp + self.owner.world.width
            self._display()
            self.location[0] = temp
        if self.wrapping["Bottom"]:
            temp = self.location[1]
            self.location[1] = temp + self.owner.world.height
            self._display()
            self.location[1] = temp
        if self.wrapping["Right"]:
            temp = self.location[0]
            self.location[0] = temp - self.owner.world.width
            self._display()
            self.location[0] = temp
        if self.wrapping["Top"]:
            temp = self.location[1]
            self.location[1] = temp - self.owner.world.height
            self._display()
            self.location[1] = temp
    
    def draw(self) -> None:
        glEnable(GL_BLEND)
        if self.scope == 0.0:
            glBegin(GL_LINES)
            glLineWidth(1.0)
            glColor4f(self.owner.colour[0], self.owner.colour[1], self.owner.colour[2], BS.SENSOR_ALPHA)
            glVertex2d(0.0, 0.0)
            glColor4f(self.owner.colour[0], self.owner.colour[1], self.owner.colour[2], BS.SENSOR_ALPHA * 2.0)
            glVertex2d(self.range, 0.0)
            glEnd()
        else:
            num_arc_pts = int(self.scope * self.range * self._beam_quality)
            angles = np.linspace(
                -0.5 * self.scope,
                0.5 * self.scope,
                num_arc_pts
            )
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(self.colour[0], self.colour[1], self.colour[2], 0.0)
            glVertex2d(0.0, 0.0)
            glColor4f(self.colour[0], self.colour[1], self.colour[2], BS.SENSOR_ALPHA)
            for angle in angles:
                glVertex2d(self.range * np.cos(angle), self.range * np.sin(angle))
            glEnd()
        glDisable(GL_BLEND)
    
    def in_scope(self, vector: Vec2) -> bool:
        if self.scope == 2 * np.pi:
            return True
        angle_to_other = get_vector_angle(vector - self.location)
        start_angle = self.orientation - 0.5 * self.scope
        if start_angle < 0:
            start_angle += 2 * np.pi
        end_angle = self.orientation + 0.5 * self.scope
        end_angle %= 2 * np.pi
        
        if start_angle < end_angle:
            return start_angle <= angle_to_other <= end_angle
        else:
            return start_angle <= angle_to_other or angle_to_other <= end_angle
