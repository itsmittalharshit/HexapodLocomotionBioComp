import numpy as np

from abc import ABC, abstractmethod, ABCMeta
from core.utils import Vec2, get_rotation_vector
from core.world.world_object import WorldObject

class SensorFunction(ABC):
    @abstractmethod
    def __call__(self):
        pass
    
    def reset(self) -> None:
        pass

class MatchFunction(SensorFunction):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, obj: WorldObject):
        pass

class EvaluateFunction(SensorFunction):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, obj: WorldObject, point: Vec2):
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        pass
    
class ScaleFunction(SensorFunction):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, value: float) -> float:
        pass

class Sensor(WorldObject):
    def __init__(
        self,
        location: Vec2 = None,
        orientation: float = None,
        relative_location: Vec2 = None,
        relative_orientation: float = 0.0,
        match_function: MatchFunction = None,
        evaluate_function: EvaluateFunction = None,
        scale_function: ScaleFunction = None
    ):
        super().__init__(location, orientation)
        assert -np.pi <= relative_orientation <= np.pi
        
        self._relative_location = relative_location
        self._relative_orientation = relative_orientation

        self.match_function = match_function
        self.evaluate_function = evaluate_function
        self.scale_function = scale_function
        self.owner: WorldObject = None
    
    def _calculate_orientation(self) -> float:
        orientation = self._relative_orientation + self.owner.orientation
        if orientation < 0:
            orientation += 2 * np.pi
        return orientation
    
    def initialise(self) -> None:
        if self._relative_location is None:
            self._relative_location = np.array([0.0, 0.0], np.float32)
        
        if self.owner is not None:
            self._start_location = self.owner.location + self._relative_location
            self._start_orientation = self._calculate_orientation()

        super().initialise()
    
    def update(self) -> None:
        self.evaluate_function.reset()
        if self.owner is not None:
            new_location = get_rotation_vector(self._relative_location, self.owner.orientation) + self.owner.location
            self.location = new_location
            self.orientation = self._calculate_orientation()
        
    def interact(self, other: WorldObject) -> None:
        if self.match_function(other):
            self.evaluate_function(other, other.location)
    
    def display(self):
        pass
    
    def output(self) -> float:
        return self.scale_function(self.evaluate_function.evaluate())
    