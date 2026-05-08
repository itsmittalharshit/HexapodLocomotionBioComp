from abc import ABC, abstractmethod
from typing import TypeVar

Genotype = TypeVar("Genotype")

class Evolver(ABC):
    def __init__(self):
        # GeneticAlgorithm attributes
        self._fitness_scores: list[float] = []
        self._probability: float = 0.0
        self._fitness: float = 0.0
        self._fixed_fitness: float = 0.0
        
        # PSOAlgorithm attributes
        self._best_solution: list[Genotype] = []
        self._best_fitness: float = 0.0
    
    @property
    def average_fitness(self):
        if len(self._fitness_scores) != 0:
            return sum(self._fitness_scores) / len(self._fitness_scores)
        else:
            return 0.0
    
    def store_fitness(self):
        self._fitness_scores.append(self.get_fitness())
    
    @abstractmethod
    def set_genotype(self, genotype: Genotype):
        pass
    
    @abstractmethod
    def get_genotype(self) -> Genotype:
        pass
    
    @abstractmethod
    def get_fitness(self):
        pass
    
    # TODO: abs method for getter?