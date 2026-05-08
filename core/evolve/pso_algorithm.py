import random

from core.evolve.base import EVO
from core.evolve.genetic_algorithm import GeneticAlgorithm

class PSOAlgorithm(GeneticAlgorithm):
    def __init__(self):
        super().__init__()
    
    def generate(self) -> None:
        self._calculate_stats()
        self._setup()
        
        for member in self.population.members:
            self.output_population.append(self.fly(member))
    
    def fly(self, evo: EVO) -> EVO:
        if not evo._best_solution or evo._fixed_fitness > evo._best_fitness:
            evo._best_solution = list(evo.get_genotype())
            evo._best_fitness = evo._fixed_fitness
        new_solution = []
        current_solution = list(evo.get_genotype())
        
        for current, p_best, g_best in zip(current_solution, evo._best_solution, self._best_current_genome):
            new_solution.append(
                current + random.uniform(0, 2) * (p_best - current) + random.uniform(0, 2) * (g_best - current)
            )
        
        new_evo: EVO = EVO()
        new_evo.set_genotype(new_solution)
        new_evo._best_solution = evo._best_solution
        new_evo._best_fitness = evo._best_fitness
        return new_evo