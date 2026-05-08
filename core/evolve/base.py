import numpy as np

from typing import TypeVar
from abc import ABC, abstractmethod
from core.evolve.evolver import Evolver
from core.world.world import World
from core.world.world_object import WorldObject

class MutationOperator(ABC):
    @abstractmethod
    def __call__(self):
        pass

class UniformMutator(MutationOperator):
    def __init__(self, minimum: float = -0.2, maximum: float = 0.2):
        self.minimum, self.maximum = minimum, maximum
    def __call__(self, t: float) -> float:
        return t + np.random.uniform(self.minimum, self.maximum)

class NormalMutator(MutationOperator):
    def __init__(self, mu: float = 0.0, sigma: float = 0.1):
        self.mu, self.sigma = mu, sigma
    def __call__(self, t: float):
        return t + np.random.normal(self.mu, self.sigma)


class SimulationObject(ABC):
    def __init__(self):
        self.world: World = None
    def begin_assessment(self):
        pass
    def end_assessment(self):
        pass
    def begin_generation(self):
        pass
    def end_generation(self):
        pass
    def begin_run(self):
        pass
    def end_run(self):
        pass
    
    @abstractmethod
    def add_to_world(self) -> None:
        pass
    
    # TODO: Serialise/unserialise? Save/load?
    
class Group(SimulationObject):
    def __init__(
        self,
        n: int,
        typing: type[WorldObject],
        *args,
        **kwargs
    ):
        self.n = n
        self.typing = typing
        self.members = [ self.typing(*args, **kwargs) for _ in range(n) ]
        
    def __del__(self):
        self.members.clear()
    
    def add_to_world(self) -> None:
        assert hasattr(self, "world")
        for m in self.members:
            self.world.add_object(m)
    
    def for_each(self, method: str, *args, **kwargs):
        return [ getattr(m, method)(*args, **kwargs) for m in self.members ]
    
    def end_assessment(self):
        for m in self.members:
            m.reset()
    
    # TODO: Serialise/unserialise?
    
Genotype = list[float]
EVO = TypeVar("evo", bound=Evolver)