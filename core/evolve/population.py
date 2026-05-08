from copy import deepcopy
from core.evolve.base import Group
from core.evolve.evolver import Evolver
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.agent.agent import Agent

class Population(Group):
    def __init__(
        self,
        population_size: int,
        population_type: type[Agent] | type[Evolver],
        genetic_algorithm: GeneticAlgorithm,
        team_size: int = -1,
        *args,
        **kwargs
    ):
        super().__init__(population_size, population_type, *args, **kwargs)
        self._genetic_algorithm = genetic_algorithm
        self._genetic_algorithm.population = self
        
        self.team_size: int = team_size
        self.num_clones: int = 0
        self.current = None
        self.team = []
        
        self.args = args
        self.kwargs = kwargs
    
    def add_to_world(self) -> None:
        assert hasattr(self, "world")
        pool = self.team if self.team_size != -1 else self.members
        for m in pool:
            self.world.add_object(m)

    
    def clone(self, member: type[Agent] | type[Evolver]) -> type[Agent] | type[Evolver]:
        return deepcopy(member)
    
    def merge(
        self,
        one: type[Agent] | type[Evolver],
        two: type[Agent] | type[Evolver]
    ) -> type[Agent] | type[Evolver]:
        one._fitness_scores.append(two.get_fitness())
        del two
        return one
    
    def average_member_fitness(self) -> list[float]:
        return [ m.average_fitness for m in self.members ]
    
    def begin_run(self) -> None:
        self.members.clear()
        self.members = [ self.typing(*self.args, **self.kwargs) for _ in range(self.n) ]
    
    def begin_generation(self) -> None:
        self.current = iter(self.members)
    
    def begin_assessment(self) -> None:
        if self.team_size != -1:
            self.team.clear()
            if self.current is None:
                self.current = iter(self.members)
                
            for _ in range(self.team_size):
                try:
                    self.team.append(next(self.current))
                except StopIteration:
                    self.current = iter(self.members)
                    self.team.append(next(self.current))
            
            for _ in range(self.num_clones):
                self.team.extend([ self.clone(m) for m in self.members ])
    
    def end_assessment(self) -> None:
        pool = self.team if self.team_size != -1 else self.members
        for m in pool:
            m.store_fitness()
            m.reset()
    
    def end_generation(self) -> None:
        self._genetic_algorithm.generate()
        self.members.clear()
        self.members.extend(self._genetic_algorithm.output_population)
        
    # TODO: Serialise/unserialise?