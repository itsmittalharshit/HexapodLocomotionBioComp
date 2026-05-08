from core.world.world_object import WorldObject
from core.agent.ffn_agent import EvolvableFFNAgent
from core.utils import ColourPalette, ColourType as CT
from core.sensor.implementation import nearest_angle_sensor
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm, GA_SELECTION_TYPE
from core.evolve.base import Group
from core.evolve.population import Population

IS_DEMO = True
DEMO_NAME = "Evo Mouse"
CLASS_NAME = "EvoMouseSimulation"

class Cheese(WorldObject):
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        self.colour = ColourPalette[CT.YELLOW]
    
    def eaten(self):
        self.location = self.world.random_location()
    
class EvoMouse(EvolvableFFNAgent):
    def __init__(self):
        super().__init__()

        self.last_cheese = 0
        self.cheese_found = 0
        sensor_range = 400

        self.add_sensor("angle", nearest_angle_sensor(Cheese, sensor_range))
        self._interaction_range = sensor_range
        self.radius = 10
        self.add_brain(10)
    
    def control(self):
        super().control()
        for key in self.controls.keys():
            self.controls[key] = self.controls[key] + 0.5
    
    def on_collision(self, obj):
        if isinstance(obj, Cheese):
            self.cheese_found += 1
            obj.eaten()

    def get_fitness(self):
        return self.cheese_found
    
    def reset(self):
        self.cheese_found = 0
        super().reset()
    
class EvoMouseSimulation(Simulation):
    def __init__(self):
        super().__init__("EvoMouse")

        self.generations = 100
        self.assessments = 1
        self.timesteps = 500

        self.population_size = 30
        self.algortihm = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        self.add("mice", Population(self.population_size, EvoMouse, self.algortihm))
        self.add("cheese", Group(self.population_size, Cheese))
    
    def log_end_generation(self):
        averages = self.contents["mice"].average_member_fitness()
        average = sum(averages) / len(averages)
        self.log.info(f"Average Fitness: {average:.3f}")
    