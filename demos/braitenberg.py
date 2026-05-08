import numpy as np

from core.world.world_object import WorldObject
from core.utils import ColourPalette, ColourType
from core.agent.agent import Agent
from core.sensor.implementation import proximity_sensor
from core.simulation import Simulation

IS_DEMO = True
DEMO_NAME = "Braitenberg"
CLASS_NAME = "BraitenbergSimulation"

class Dot(WorldObject):
    def __init__(self, l):
        super().__init__(l, 0.0, 12.5)
        self.colour = ColourPalette[ColourType.YELLOW]
    
    def __del__(self):
        pass
    
class Braitenberg(Agent):
    def __init__(self):
        super().__init__(random_colour=False)
        
        self.add_sensor("left", proximity_sensor(Dot, np.pi/2, 125, np.pi/2.5, True, maximum=0.5))
        self.add_sensor("right", proximity_sensor(Dot, np.pi/2, 125, -np.pi/2.5, True, maximum=0.5))
        self.sensors["left"].draw_fixed = True
        self.sensors["right"].draw_fixed = True
        
        self._min_speed = 20.0
        self._max_speed = 90.0
        self._timestep = 0.05
        self.radius = 10.0
        self._max_rotate = 2 * np.pi
        
    def control(self):
        self.controls["left"] = self.sensors["right"].output()
        self.controls["right"] = self.sensors["left"].output()

class Braitenberg2a(Braitenberg):
    def __init__(self):
        super().__init__()
        self.colour = ColourPalette[ColourType.RED]
    
    def control(self):
        self.controls["left"] = self.sensors["left"].output()
        self.controls["right"] = self.sensors["right"].output()

class Braitenberg2b(Braitenberg):
    def __init__(self):
        super().__init__()
        self.colour = ColourPalette[ColourType.BLUE]
    
    def control(self):
        self.controls["left"] = self.sensors["right"].output()
        self.controls["right"] = self.sensors["left"].output()
        
class BraitenbergSimulation(Simulation):
    def __init__(self):
        super().__init__("Braitenberg")
        self.timesteps = -1
    
    def begin_assessment(self):
        self.world.add_object(Braitenberg2a())
        self.world.add_object(Braitenberg2b())
        
        positions = [(150.0, 100.0), (200.0, 100.0), (250.0, 100.0), (300.0, 100.0),
                     (350.0, 100.0), (350.0, 150.0), (350.0, 200.0),
                     (350.0, 250.0), (350, 300.0), (350.0, 350.0),
                     (300.0, 350.0), (250.0, 350.0), (200.0, 350.0), (200.0, 400.0),
                     (200.0, 450.0), (200.0, 500.0), (200.0, 550.0), (250.0, 550.0),
                     (300.0, 550.0), (350.0, 550.0), (400.0, 550.0), (450.0, 550.0),
                     (500.0, 550.0), (550.0, 550.0), (600.0, 550.0), (600.0, 500.0),
                     (600.0, 450.0), (600.0, 400.0), (600.0, 350.0), (550.0, 350.0),
                     (500.0, 350.0), (500.0, 300.0), (500.0, 250.0), (500.0, 200.0),
                     (500.0, 150.0), (500.0, 100.0), (500.0, 50.0)]
        
        for pos in positions:
            self.world.add_object(Dot(pos))
        
        super().begin_assessment()