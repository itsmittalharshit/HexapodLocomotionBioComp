import random

from core.sensor.base import ScaleFunction

class ScaleCompose:
    def __init__(
        self,
        first: ScaleFunction,
        second: ScaleFunction
    ):
        self.functions = [first, second]
    
    def __call__(self, value: float) -> float:
        return self.second(self.first(value))

class ScaleLinear(ScaleFunction):
    def __init__(
        self,
        input_min: float,
        input_max: float,
        output_min: float = 0.0,
        output_max: float = 1.0
    ):
        self.input_min, self.input_max = input_min, input_max
        self.output_min, self.output_max = output_min, output_max
    
    def __call__(self, value: float) -> float:
        return (value - self.input_min) / (self.input_max - self.input_min) * (self.output_max - self.output_min) + self.output_min

class ScaleAbsolute(ScaleFunction):
    def __call__(self, value: float) -> float:
        return value if value >= 0.0 else -value

class ScaleThreshold(ScaleFunction):
    def __init__(
        self,
        threshold: float,
        minimum: float = 0.0,
        maximum: float = 1.0
    ):
        self.threshold, self.minimum, self.maximum = threshold, minimum, maximum
    
    def __call__(self, value: float) -> float:
        return self.minimum if value < self.threshold else self.maximum
    
class ScaleNoise(ScaleFunction):
    def __init__(self, minimum: float = -0.1, maximum: float = 0.1):
        self.minimum, self.maximum = minimum, maximum
    
    def __call__(self, value: float) -> float:
        return value + random.uniform(self.minimum, self.maximum)

class ScaleAdapter(ScaleFunction):
    def __init__(self, function):
        self.function = function
    
    def __call__(self, value):
        return self.function(value)
