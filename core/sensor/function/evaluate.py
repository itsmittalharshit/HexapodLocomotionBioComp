import numpy as np

from core.sensor.base import EvaluateFunction
from core.sensor.beam_sensor import BeamSensor
from core.world.world_object import WorldObject
from core.utils import Vec2, get_vector_angle, WORLD_DISPLAY_PARAMETERS

class EvaluateNearest(EvaluateFunction):
    def __init__(
        self,
        owner: WorldObject,
        sensor_range: float
    ):
        self.range = sensor_range
        self.owner = owner
        
        self.nearest_so_far = sensor_range
        self.best_candidate = None
        self.best_candidate_vector: Vec2 = None
        self.distance: float = 0.0
        self.threshold: float = 1.0
    
    def reset(self) -> None:
        self.best_candidate = None
        self.best_candidate_vector = None
        self.nearest_so_far = self.range
    
    def __call__(self, obj: WorldObject, loc: Vec2):
        self.distance = np.linalg.norm(self.owner.location - loc)
        if self.distance < self.nearest_so_far:
            self.nearest_so_far = self.distance
            self.best_candidate = obj
            self.best_candidate_vector = loc
    
    def evaluate(self) -> float:
        return self.nearest_so_far

class EvaluateNearestInScope(EvaluateNearest):
    def __init__(
        self,
        owner: BeamSensor,
        scope: float,
        sensor_range: float
    ):
        self.scope = scope
        super().__init__(owner, sensor_range)
    
    def __call__(self, obj: WorldObject, loc: Vec2):
        if self.scope == 2 * np.pi:
            super().__call__(obj, loc)
        elif self.owner.in_scope(loc):
            super().__call__(obj, loc)

class EvaluateBeam(EvaluateNearestInScope):
    def __call__(self, obj: WorldObject, loc: Vec2):
        self.evaluate(obj)
        self.owner: BeamSensor
        if not self.owner.wrap:
            return
        if self.owner.wrapping["Left"]:
            temp = self.owner.location[0]
            self.owner.location[0] = temp + self.owner.world._display_params.width
            self.evaluate(obj)
            self.owner.location[0] = temp
        if self.owner.wrapping["Bottom"]:
            temp = self.owner.location[1]
            self.owner.location[1] = temp + self.owner.world._display_params.height
            self.evaluate(obj)
            self.owner.location[0] = temp
        if self.owner.wrapping["Right"]:
            temp = self.owner.location[0]
            self.owner.location[0] = temp - self.owner.world._display_params.width
            self.evaluate(obj)
            self.owner.location[0] = temp
        if self.owner.wrapping["Top"]:
            temp = self.owner.location[0]
            self.owner.location[1] = temp - self.owner.world._display_params.width
            self.evaluate(obj)
            self.owner.location[1] = temp

class EvaluateNearestDistanceX(EvaluateNearest):
    """
    Returns vertical distance to nearest target
    """
    def evaluate(self) -> float:
        if self.best_candidate_vector is None:
            return 0.0
        
        delta = self.best_candidate_vector[0] - self.owner.location[0]
        if abs(delta) > (WORLD_DISPLAY_PARAMETERS.width / 2):
            score = abs(WORLD_DISPLAY_PARAMETERS.width - abs(delta))
        else:
            score = abs(delta)
        
        if score > self.range:
            self.reset()
            return 0.0
        else:
            return score
        
class EvaluateNearestDistanceY(EvaluateNearest):
    """
    Returns horizontal distance to nearest target
    """
    def evaluate(self) -> float:
        if self.best_candidate_vector is None:
            return 0.0
        
        delta = self.best_candidate_vector[1] - self.owner.location[1]
        if abs(delta) > (WORLD_DISPLAY_PARAMETERS.height / 2):
            score = abs(WORLD_DISPLAY_PARAMETERS.height - abs(delta))
        else:
            score = abs(delta)
        
        if score > self.range:
            self.reset()
        else:
            return score

class EvaluateNearestPositionX(EvaluateNearest):
    def evaluate(self) -> float:
        if self.best_candidate_vector is None:
            return 0.0
        return self.best_candidate_vector[0]

class EvaluateNearestPositionY(EvaluateNearest):
    def evaluate(self) -> float:
        if self.best_candidate_vector is None:
            return 0.0
        return self.best_candidate_vector[1]

class EvaluateNearestAngle(EvaluateNearest):
    def evaluate(self) -> float:
        if self.best_candidate is None:
            return 0.0
        else:
            angle = get_vector_angle(self.best_candidate_vector - self.owner.location)
            if angle > np.pi:
                angle -= 2 * np.pi
            return angle

class EvaluateCount(EvaluateFunction):
    def __init__(self, start: int = 0):
        self.start = start
        self.count: int = 0
    
    def reset(self) -> None:
        self.count = 0
    
    def __call__(self, obj: WorldObject, loc: Vec2):
        self.count += 1
    
    def evaluate(self) -> float:
        return float(self.count + self.start)

class EvaluateProximity(EvaluateFunction):
    def __init__(
        self,
        owner: WorldObject,
        sensor_range: float,
        n_max: int = 3
    ):
        self.owner, self.range, self.n_max = owner, sensor_range, n_max
        self.distances = []
    
    def reset(self) -> None:
        self.distances.clear()
    
    def __call__(self, obj: WorldObject, loc: Vec2):
        distance = np.linalg.norm(self.owner.location - loc)
        if distance < self.range:
            self.distances.append(distance)
    
    def evaluate(self) -> float:
        if len(self.distances) == 0:
            return 0
        
        if len(self.distances) > self.n_max:
            distances = np.sort(self.distances)
            distances = distances[:self.n_max]
        else:
            distances = self.distances
        
        proximity_scores = np.zeros(len(distances))
        for i, distance in enumerate(distances):
            proximity_scores[i] = 1.0 / (1 + np.abs(np.log10(distance)))
        return np.sum(proximity_scores)
