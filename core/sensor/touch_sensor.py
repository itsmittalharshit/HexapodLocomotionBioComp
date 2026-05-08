from core.sensor.base import Sensor
from core.world.world_object import WorldObject

class TouchSensor(Sensor):
    def initialise(self) -> None:
        if self.owner:
            self.radius = self.owner.radius
        super().initialise()
    
    def interact(self, other: WorldObject) -> None:
        if self.match_function and self.owner and self.owner.is_touching(other):
            self.evaluate_function(other, other.nearest_point(self))
    