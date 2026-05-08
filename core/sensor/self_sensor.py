import numpy as np

from core.sensor.base import Sensor

class SelfSensor(Sensor):
    def __init__(
        self,
        typing: str = "X",
        control: str = ""
    ):
        super().__init__()
        assert typing in ["X", "Y", "Angle", "Control"]
        self.typing = typing
        self.control = control
    
    def output(self) -> float:
        if self.owner is None:
            return 0.0
        if self.typing == "X":
            return self.owner.location[0] / self.owner.world.width
        elif self.typing == "Y":
            return self.owner.location[1] / self.owner.world.height
        elif self.typing == "Angle":
            return self.owner.orientation / (2 * np.pi)
        elif self.typing == "Control":
            return self.owner.controls.get(self.control, 0.0)
        
        return 0.0