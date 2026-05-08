import numpy as np

from core.agent.ffn_agent import EvolvableFFNAgent
from core.evolve.evolver import Evolver
from core.sensor.implementation import proximity_sensor
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.utils import GA_SELECTION_TYPE
from core.evolve.population import Population

IS_DEMO = True
DEMO_NAME = "Chase"
CLASS_NAME = "ChaseSimulation"

class Prey(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)
        
        self.times_eaten = 0
        range = 300.0
        
        self.add_sensor("left", proximity_sensor(Predator, np.pi/4, range, np.pi/8, True))
        self.add_sensor("right", proximity_sensor(Predator, np.pi/4, range, -np.pi/8, True))
        self._interaction_range = range
        
        self.add_brain(4)
        
        self.solid = False
        self.radius = 10
        self._min_speed = 0.0
        self._max_speed = 100.0
    
    def control(self):
        super().control()
        for k in self.controls.keys():
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)
    
    def eaten(self):
        self.times_eaten += 1
        self.location = self.world.random_location()
        self.trail.clear()
    
    def get_fitness(self):
        if self.times_eaten == 0:
            return 1.0
        return 1.0 / self.times_eaten
    
    def reset(self):
        self.times_eaten = 0
    

class Predator(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)
        
        self.prey_eaten = 0
        range = 300.0
        
        self.add_sensor("left", proximity_sensor(Prey, np.pi/4, range, np.pi/8, True))
        self.add_sensor("right", proximity_sensor(Prey, np.pi/4, range, -np.pi/8, True))
        self._interaction_range = range
        
        self.add_brain(4)
        self.solid = False
        self._min_speed = 0.0
        self._max_speed = 110.0
        self.radius = 20.0
        
    def control(self):
        super().control()
        for k in self.controls.keys():
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)
    
    def on_collision(self, other):
        if isinstance(other, Prey):
            self.prey_eaten += 1
            other.eaten()
        super().on_collision(other)
    
    def get_fitness(self):
        return self.prey_eaten
    
    def reset(self):
        self.prey_eaten = 0

class ChaseSimulation(Simulation):
    def __init__(self):
        super().__init__("Chase")
        
        self.runs = 1
        self.generations = 2000
        self.assessments = 3
        self.timesteps = 500
        
        population_prey, population_pred = 30, 30
        ga_prey = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        ga_pred = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        
        self.add("prey", Population(population_prey, Prey, ga_prey, team_size=10))
        self.add("predator", Population(population_pred, Predator, ga_pred, team_size=10))

    def log_end_generation(self):
        prey_averages = self.contents["prey"].average_member_fitness()
        prey_average = sum(prey_averages) / len(prey_averages)
        self.log.info(f"Average prey Fitness: {prey_average:.3f}")

        predator_averages = self.contents["predator"].average_member_fitness()
        predator_average = sum(predator_averages) / len(predator_averages)
        self.log.info(f"Average predator Fitness: {predator_average:.3f}")
