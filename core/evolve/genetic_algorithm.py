import random
import numpy as np

from abc import ABC
from copy import deepcopy
from core.utils import GA_SELECTION_TYPE, GA_FITNESS_METHOD, GA_FITNESS_FIX, GA_PRINT_TYPE, GA_FLOAT_DEFAULT, GA_INT_DEFAULT
from core.evolve.base import MutationOperator, NormalMutator, Genotype, EVO

class GeneticAlgorithm(ABC):
    def __init__(
        self,
        crossover: float = 0.7,
        mutation: float = 0.05,
        crossover_points: int = 1,
        team_size: int = -1,
        selection: int = GA_SELECTION_TYPE.RANK,
        fitness_method: int = GA_FITNESS_METHOD.BEST,
        fitness_fix: int = GA_FITNESS_FIX.IGNORE,
        print_style: list[int] = [GA_PRINT_TYPE.PARAMETERS],
        elitism: int = 0,
        culling: int = 0,
        mutator = None
    ):
        assert culling % 2 == 0
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_points = crossover_points
        self.elitism = elitism
        self.culling = culling
        self.team_size = team_size
        self.output_population = []
        
        self.selection = selection
        self.fitness_method = fitness_method
        self.fitness_fix = fitness_fix
        
        self._float_params = GA_FLOAT_DEFAULT
        self._int_params = GA_INT_DEFAULT
        
        if mutator is None:
            self.mutator = NormalMutator()
        else:
            self.mutator = mutator
        
        self.print_style = print_style
        
        self.owns_data: bool = False
        
        self.input_population_size: int = 0
        self.output_population_size: int = 0
        self.chromosome_length = 0
        self.total_fitness: float = 0.0
        self.best_fitness: float = 0.0
        self.total_fixed_fitness: float = 0.0
        self.worst_fitness: float = 0.0
        self.total_probability: float =0.0
        
        self.generations: int = 0
        self._average_fitness_record: list[float] = []
        self._best_fitness_record: list[float] = []
        self._best_ever_fitness: float = 0.0
        self._best_ever_genome = None
        self._best_current_genome = None
        
        self.population = None
    
    # TODO: Destructor?
    # TODO: __str__?
    
    def generate(self) -> None:
        self._calculate_stats()
        self._setup()
        
        if self.culling > 0:
            for _ in range(self.culling):
                self.population.members.pop()
        if self.elitism > 0:
            for evo in self.population.members[:self.elitism]:
                new_evo = self.add_member()
                new_evo.initialise()
                new_evo.set_genotype(evo.get_genotype())
                self.output_population.append(new_evo)
        
        for _ in range((self.output_population_size - self.elitism) // 2):
            mother = self.select_parent_genotype()
            father = self.select_parent_genotype()
            for _ in range(self.crossover_points):
                if random.random() < self.crossover:
                    mother, father = self.crossover_genotypes(mother, father)
            self.mutate_genotype(mother)
            self.mutate_genotype(father)
            
            evo1 = self.add_member()
            evo2 = self.add_member()
            evo1.set_genotype(mother)
            evo2.set_genotype(father)
            self.output_population.append(evo1)
            self.output_population.append(evo2)
                
    def _calculate_stats(self) -> None:
        self.input_population_size = len(self.population.members)
        assert self.input_population_size % 2 == 0
        assert self.culling <= self.input_population_size - 2
        assert self.elitism <= self.input_population_size - self.culling
        
        best_evo_so_far = self.population.members[0]
        self.best_fitness = self.worst_fitness = self.get_fitness(best_evo_so_far)
        self.total_fitness = 0.0
        
        for evo in self.population.members:
            f = self.get_fitness(evo)
            if f is None:
                continue
            elif f > self.best_fitness:
                best_evo_so_far = evo
                self.best_fitness = f
            elif f < self.worst_fitness:
                self.worst_fitness = f
            self.total_fitness += f
        
        self.generations += 1
        self._average_fitness_record.append(self.total_fitness / float(self.input_population_size))
        self._best_fitness_record.append(self.best_fitness)
        self._best_current_genome = best_evo_so_far.get_genotype()
        
        if self.best_fitness > self._best_ever_fitness:
            self._best_ever_fitness = self.best_fitness
            self._best_ever_genome = self._best_current_genome
    
    def _setup(self) -> None:
        self.output_population.clear()
        self.output_population_size = len(self.population.members)
        
        # TODO: chromosome length to shortest of any given pair?
        self.chromosome_length = len(self.population.members[0].get_genotype())
        
        self.fix_fitness()
        self.population.members.sort(key=lambda x: x._fixed_fitness, reverse=True)
        self.total_probability = 0
        
        if self.selection == GA_SELECTION_TYPE.ROULETTE:
            for evo in self.population.members:
                if evo._fitness is None:
                    evo._probability = 0.0
                else:
                    evo._probability = (evo._fixed_fitness / self.total_fixed_fitness) ** self._float_params.EXPONENT
                    self.total_probability += evo._probability
        if self.selection == GA_SELECTION_TYPE.RANK:
            for rank, evo in enumerate(self.population.members):
                if evo._fitness is None:
                    evo._probability = 0.0
                else:
                    evo._probability = (1.0 - rank / (self.input_population_size - 1)) ** self._float_params.EXPONENT
                    self.total_probability += evo._probability
        elif self.selection in [GA_SELECTION_TYPE.ROULETTE, GA_SELECTION_TYPE.RANK]:
            for evo in self.population.members:
                evo._probability /= self.total_probability
        
    def clean(self) -> None:
        if self.owns_data:
            self.population.clear()
            self.output_population.clear()
        else:
            self.population = []
            self.output_population = []
    
    def add_member(self) -> EVO:
        return self.population.typing(*self.population.args, **self.population.kwargs)
    
    def fix_fitness(self) -> None:
        """
        Normalise according to self.fitness_fix
        """
        total_fixed_fitness = 0
        for evo in self.population.members:
            f = evo._fitness
            if f is None:
                continue
            elif self.fitness_fix == GA_FITNESS_FIX.FIX:
                f -= self.worst_fitness
            elif self.fitness_fix == GA_FITNESS_FIX.CLAMP:
                f = max(0, f)
            elif self.fitness_fix == GA_FITNESS_FIX.IGNORE:
                pass
            evo._fixed_fitness = f
            total_fixed_fitness += f
        self.total_fixed_fitness = total_fixed_fitness
    
    def select_parent_genotype(self) -> Genotype:
        if self.selection in [GA_SELECTION_TYPE.ROULETTE, GA_SELECTION_TYPE.RANK]:
            chromo = self.select_probability()
        elif self.selection == GA_SELECTION_TYPE.TOURNAMENT:
            chromo = self.select_tournament()
        return chromo
    
    def select_probability(self) -> Genotype:
        probabilities = [ evo._probability for evo in self.population.members ]
        evo = random.choices(self.population.members, weights=probabilities, k=1)[0]
        return evo.get_genotype()
    
    def select_tournament(self) -> Genotype:
        tournament = random.sample(self.population.members, self._int_params.TOURNAMENT_SIZE)
        if random.random() < self._float_params.TOURNAMENT_PARAM:
            winner = max(tournament, key=lambda evo: evo._fixed_fitness)
        else:
            winner = random.choice(tournament)
        return winner.get_genotype()
    
    def crossover_genotypes(self, mother: Genotype, father: Genotype):
        crossover_point = random.randint(0, self.chromosome_length - 1)
        child1 = np.concatenate((mother[:crossover_point], father[crossover_point:]))
        child2 = np.concatenate((father[:crossover_point], mother[crossover_point:]))
        return child1, child2
    
    def mutate_genotype(self, genome: Genotype):
        for i in range(len(genome)):
            if random.random() < self.mutation:
                genome[i] = self.mutator(genome[i])
    
    def get_fitness(self, evo: EVO) -> float:
        if not evo._fitness_scores:
            return None
        value = self._calculate_fitness(evo)
        evo._fitness = value
        return value
    
    def _calculate_fitness(self, evo: EVO) -> float:
        if self.fitness_method == GA_FITNESS_METHOD.BEST:
            return np.max(evo._fitness_scores)
        elif self.fitness_method == GA_FITNESS_METHOD.WORST:
            return np.min(evo._fitness_scores)
        elif self.fitness_method == GA_FITNESS_METHOD.MEAN:
            return np.mean(evo._fitness_scores)
        elif self.fitness_method == GA_FITNESS_METHOD.TOTAL:
            return np.sum(evo._fitness_scores)
        else:
            return 0.0
    
    def copy_population(self) -> list[EVO]:
        return deepcopy(self.output_population)
    
    def set_crossover(self, c: float) -> None:
        assert 0 <= c <= 1.0
        self.crossover = c
    
    def set_mutation(self, m: float) -> None:
        assert 0 <= m <= 1.0
        self.mutation = m
    
    def set_elitism(self, e: int) -> None:
        assert 0 <= e <= self.input_population_size
        self.elitism = e
    
    def set_culling(self, c: int) -> None:
        assert 0 <= c <= self.input_population_size
        self.culling = c
    
    def set_crossover_points(self, p: int) -> None:
        assert p >= 0
        self.crossover_points = 0
    
    def set_fitness_method(self, f: str) -> None:
        assert f in GA_FITNESS_METHOD
        self.fitness_method = getattr(GA_FITNESS_METHOD, f)
    
    def set_fitness_fix(self, f: str) -> None:
        assert f in GA_FITNESS_FIX
        self.fitness_fix = getattr(GA_FITNESS_FIX, f)
    
    def get_csv(self, separator: str = ','):
        # TODO
        pass
    
    # TODO: Serialise/unserialise?