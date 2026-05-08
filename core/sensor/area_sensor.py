from core.sensor.base import Sensor
from core.world.world_object import WorldObject

class AreaSensor(Sensor):
    def interact(self, other: WorldObject) -> None:
        vector, _ = other.nearest_point(self)
        if self.match_function and self.evaluate_function and self.is_inside(vector):
            self.evaluate_function(other, vector)