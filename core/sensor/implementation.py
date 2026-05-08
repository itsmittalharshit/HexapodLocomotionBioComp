import numpy as np

from core.sensor.base import Sensor
from core.sensor.beam_sensor import BeamSensor
from core.sensor.touch_sensor import TouchSensor

from core.sensor.function.evaluate import *
from core.sensor.function.match import *
from core.sensor.function.scale import *

from core.world.world_object import WorldObject

def proximity_sensor(
    typing: type[WorldObject],
    scope: float,
    sensor_range: float,
    orientation: float,
    simple: bool = False,
    minimum = 1.0,
    maximum = 0.0
) -> Sensor:
    if not simple:
        s = BeamSensor(scope, sensor_range, relative_orientation = orientation)
        s.evaluate_function = EvaluateBeam(s, scope, sensor_range)
    else:
        s = BeamSensor(scope, sensor_range, relative_orientation = orientation)
        s.evaluate_function = EvaluateNearestInScope(s, scope, sensor_range)
    
    s.match_function = MatchKind(typing)
    s.scale_function = ScaleLinear(0.0, sensor_range, minimum, maximum)
    return s

def nearest_angle_sensor(
    typing: type[WorldObject],
    sensor_range: float = 1000.0,
    reverse = False
) -> Sensor:
    s = Sensor()
    s.match_function = MatchKind(typing)
    s.evaluate_function = EvaluateNearestAngle(s, sensor_range)
    if not reverse:
        s.scale_function = ScaleLinear(-np.pi, np.pi, -1.0, 1.0)
    else:
        s.scale_function = ScaleLinear(-np.pi, np.pi, 1.0, -1.0)
    return s

def nearest_x_sensor(typing: type[WorldObject], sensor_range: float = 1000.0) -> Sensor:
    s = Sensor(np.array([0, 0], np.float32), 0.0)
    s.match_function = MatchKind(typing)
    s.evaluate_function = EvaluateNearestDistanceX(s, sensor_range)
    s.scale_function = ScaleLinear(0, sensor_range, -1.0, 1.0)

def nearest_y_sensor(typing: type[WorldObject], sensor_range: float = 1000.0) -> Sensor:
    s = Sensor(np.array([0, 0], np.float32), 0.0)
    s.match_function = MatchKind(typing)
    s.evaluate_function = EvaluateNearestDistanceY(s, sensor_range)
    s.scale_function = ScaleLinear(0, sensor_range, -1.0, 1.0)

def density_sensor(
    typing: type[WorldObject],
    scope: float,
    sensor_range: float,
    orientation: float
) -> Sensor:
    s = BeamSensor(scope, sensor_range, np.array([0, 0], np.float32), orientation)
    s.draw_fixed = True
    s.match_function = MatchKind(typing)
    s.evaluate_function = EvaluateCount(1)
    s.scale_function = ScaleCompose(
        ScaleAdapter(lambda f: 1.0 / f),
        ScaleLinear(0.0, 1.0, 1.0, 0.0)
    )
    return s

def collision_sensor(typing: type[WorldObject], threshold: float = 1.0) -> Sensor:
    s = TouchSensor()
    s.match_function = MatchKind(typing)
    s.evaluate_function = EvaluateCount()
    s.scale_function = ScaleThreshold(1.0)
    return s
