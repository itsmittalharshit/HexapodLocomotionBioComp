from core.agent.agent import Agent
from core.simulation import Simulation
from core.evolve.base import Group
from core.world.world_object import WorldObject
from core.utils import ColourPalette, ColourType as CT
from core.sensor.implementation import nearest_angle_sensor

IS_DEMO = True
DEMO_NAME = "Mouse"
CLASS_NAME = "MouseSimulation"

class Cheese(WorldObject):
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        self.colour = ColourPalette[CT.YELLOW]
    
    def eaten(self):
        self.location = self.world.random_location()

class Mouse(Agent):
    def __init__(self):
        super().__init__()

        self.cheese_found = 0
        sensor_range = 250
        
        self.add_sensor("angle", nearest_angle_sensor(Cheese, sensor_range))
        self._interaction_range = sensor_range
        self._min_speed = 25.0
        self._max_speed = 100.0

        self.radius = 10
    
    def reset(self):
        self.cheese_found = 0
        super().reset()
    
    def on_collision(self, other):
        if isinstance(other, Cheese):
            self.cheese_found += 1
            other.eaten()

class MouseSimulation(Simulation):
    def __init__(self):
        super().__init__("Mouse")
        self.assessments = 2
        self.timesteps = 1000
        self.generations = 200

        mice = Group(30, Mouse)
        cheeses = Group(30, Cheese)
        
        self.add("mice", mice)
        self.add("cheeses", cheeses)