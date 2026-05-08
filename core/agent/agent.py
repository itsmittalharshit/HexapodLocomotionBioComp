import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from core.world.drawable import Drawable
from core.sensor.base import Sensor
from core.world.world_object import WorldObject
from core.world.trail import Trail
from core.utils import Vec2, AgentSettings as AS, AGENT_COLOURS, AgentPart, length_angle_to_vector, random_colour, normalise_vector, get_reciprocal

class Agent(WorldObject):
    __metaclass__ = ABCMeta
    _count = 0
    
    def __init__(
        self,
        location: Vec2 = None,
        orientation: Vec2 = None,
        velocity: Vec2 = None,
        min_speed: float = AS.MIN_SPEED,
        max_speed: float = AS.MAX_SPEED,
        max_rotate: float = AS.MAX_ROTATE,
        timestep: float = AS.TIMESTEP,
        solid: bool = False,
        random_colour: bool = True,
        interaction_range: float = np.inf,
        controls: dict[str, float] = None
    ):
        super().__init__(
            location,
            orientation,
            AS.RADIUS,
            solid = solid
        )

        self._start_velocity = velocity
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._max_rotate = max_rotate
        self._timestep = timestep
        self._random_colour = random_colour
        self._interaction_range = interaction_range
        self._colours = AGENT_COLOURS
        self._colours[AgentPart.BODY] = self.colour
        
        self._collision_point: Vec2 = None
        self._collision_normal: Vec2 = None
        
        self.initialised = False
        
        self.distance_travelled: float = 0.0
        self.power_used: float = 0.0
        self.sensors : dict[str, Sensor] = {}
        if controls is None:
            self.controls = {
                "left": 0.0,
                "right": 0.0
            }
        self.trail = Trail()
        self._reset_random["velocity"] = True if velocity is None else False
        
        Agent._count += 1
    
    def __del__(self):
        Agent._count -= 1
        for _, sensor in self.sensors.items():
            if sensor.owner == self:
                del sensor
        super().__del__()
    '''
    def __repr__(self):
        if not self.initialised:
            return self._repr(
                initialised = self.initialised,
                start_location = self._start_location,
                start_orientation = self._start_orientation,
            )
        else:
            return self._repr(
                initialised = self.initialised,
                location = self.location,
                orientation = self.orientation,
                velocity = self.velocity,
                colour = self.colour,
                edges = self.edges
            )
    '''
    def initialise(self) -> None:
        super().initialise()
        if self._start_velocity is None:
            self._start_velocity = length_angle_to_vector(1.0, self.orientation) 
        self.velocity = self._start_velocity

        self._initialise_colour()
        self.trail.colour = [self.colour[0], self.colour[1], self.colour[2]]
        
        for sensor in self.sensors.values():
            sensor.initialise()
        
        self.initialised = True
    
    def _initialise_colour(self) -> None:
        if self._random_colour:
            self.colour = random_colour()
        self._colours[AgentPart.BODY][:] = self.colour
        self._colours[AgentPart.ARROW][:] = self.colour
    
    def add_sensor(self, name: str, s: Sensor) -> None:
        if name in self.sensors and self.sensors[name].owner == self:
            del self.sensors[name]
        self.sensors[name] = s
        s.owner = self
    
    def update(self) -> None:
        self.control()
        dt = self._timestep
        if callable(self.controls["left"]):
            control_left = self.controls["left"]()
        elif isinstance(self.controls["left"], float):
            control_left = self.controls["left"]
        else:
            assert False
        if callable(self.controls["right"]):
            control_right = self.controls["right"]()
        elif isinstance(self.controls["right"], float):
            control_right = self.controls["right"]
        else:
            assert False
        
        self.offset_orientation(self._max_rotate * (control_left - control_right) * dt)
        self.velocity += length_angle_to_vector(
            (self._max_speed - self._min_speed) * 0.5 * (control_left + control_right) + self._min_speed,
            self.orientation
        )
        
        # 150 * 0.5 * (co)
        
        if self._max_speed > 0.0:
            self.velocity -= self.velocity * (1.0 / self._max_speed) * AS.DRAG
        
        if np.linalg.norm(self.velocity)**2 > self._max_speed**2:
            self.velocity = normalise_vector(self.velocity) * self._max_speed
        self.location += self.velocity * dt
        
        # Clear trails whilst agent transported during update
        while self.location[0] < 0:
            self.location[0] = self.location[0] + self.world._display_params.width
            self.trail.clear()
        while self.location[0] >= self.world._display_params.width:
            self.location[0] = self.location[0] - self.world._display_params.width
            self.trail.clear()
        while self.location[1] < 0:
            self.location[1] = self.location[1] + self.world._display_params.height
            self.trail.clear()
        while self.location[1] >= self.world._display_params.height:
            self.location[1] = self.location[1] - self.world._display_params.height
            self.trail.clear()
        
        for sensor in self.sensors.values():
            sensor.update()
        
        self.distance_travelled += np.linalg.norm(self.velocity) * dt

        for control in self.controls.values():
            self.power_used += ((self._max_speed - self._min_speed) * abs(control) + self._min_speed)
        
        self.trail.append(deepcopy(self.location))
        self.trail.update()
        super().update()
    
    def reset(self):
        super().reset()
        self.distance_travelled = 0.0
        self.power_used = 0.0
        if self._reset_random["velocity"]:
            self._start_velocity = length_angle_to_vector(1.0, self.orientation)
        self.velocity = self._start_velocity
        self.trail.clear()
    
    # @abstractmethod
    def control(self):
        pass
    
    def interact(self, other: WorldObject) -> None:
        if np.linalg.norm(self.location - other.location) <= self._interaction_range:
            if isinstance(other, Agent):
                self.sensor_interact(other)
            
            if self.is_touching(other):
                if self.solid and other.solid:
                    if hasattr(other, "velocity"):
                        ov = other.velocity
                    else:
                        ov = 0.0
                    average_velocity = (self.velocity + ov) * 0.5
                    vec_to_other = other.location - self.location
                    min_distance = self.radius + other.radius
                    self.velocity = average_velocity
                    if hasattr(other, "velocity"):
                        other.velocity = average_velocity
                    
                    self.location += normalise_vector(get_reciprocal(vec_to_other)) * (min_distance - np.linalg.norm(vec_to_other))
                    if hasattr(other, "velocity"):
                        other.location += normalise_vector(vec_to_other) * (min_distance - np.linalg.norm(vec_to_other))
                self.on_collision(other)
                other.on_collision(self)
                self.world.add_collision(self._collision_point)
            else:
                self.sensor_interact(other)
                if self.is_touching(other):
                    self.location += self._collision_normal * (self.radius - np.linalg.norm(self.location - self._collision_point))
                    self.on_collision(other)
                    other.on_collision(self)
                    self.world.add_collision(self._collision_point)
        super().interact(other) # WorldObject does not implement interact
    
    def is_touching(self, other: WorldObject) -> bool:
        vec_to_other = other.location - self.location
        min_distance = self.radius + other.radius
        if np.linalg.norm(vec_to_other)**2 > min_distance**2:
            return False
        self._collision_point, self._collision_normal = other.nearest_point(self.location)
        return other.circular or self.is_inside(self._collision_point)
    
    
    def sensor_interact(self, other: WorldObject) -> None:
        for s in self.sensors.values():
            s.interact(other)
    
    def display(self) -> None:
        if self.world._display_params.config & self.world._display_type.DISPLAY_SENSORS != 0:
            for s in self.sensors.values():
                s.display()
        if self.world._display_params.config & self.world._display_type.DISPLAY_TRAILS != 0:
            try:
                self.trail.display()
            except:
                pass
        if self.world._display_params.config:
            super().display()
    
    def draw(self) -> None:
        # TODO: Weird extra code skipped
        temp_color = self.colour
        Drawable.draw(self)
        self.colour = temp_color
        
        # Centre
        glColor4fv(self._colours[AgentPart.CENTRE])
        disk = gluNewQuadric()
        gluQuadricDrawStyle(disk, GLU_FILL)
        gluDisk(disk, 0, (self.radius / 0.85) - 4.0, 20, 1)
        gluDeleteQuadric(disk)
        
        # Arrow
        glColor4fv(self._colours[AgentPart.ARROW])
        glLineWidth(1.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(0.0, self.radius / 2.0)
        glVertex2d(self.radius / 1.5, 0.0)
        glVertex2d(0.0, self.radius / -2.0)
        glEnd()
        
        # Right Wheel
        glColor4fv(self._colours[AgentPart.WHEEL])
        glLineWidth(4.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(self.radius / -2.0, 2.0 - self.radius)
        glVertex2d(self.radius / 2.0, 2.0 - self.radius)
        glEnd()
        
        # Left Wheel
        glColor4fv(self._colours[AgentPart.WHEEL])
        glLineWidth(4.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(self.radius / -2.0, self.radius - 2.0)
        glVertex2d(self.radius / 2.0, self.radius - 2.0)
        glEnd()
    
    # TODO: Serialise/Unserialise?
