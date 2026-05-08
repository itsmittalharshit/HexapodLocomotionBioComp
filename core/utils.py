import numpy as np

from typing import TypeAlias, Literal
from types import SimpleNamespace

# Vector Type Aliases
Vec2: TypeAlias = np.ndarray[tuple[Literal[2],], np.dtype[np.float32]]
Vec3: TypeAlias = np.ndarray[tuple[Literal[3],], np.dtype[np.float32]]
###

# Vector Maths
def get_rotation_vector(vector: Vec2, angle: float) -> Vec2:
    m1 = np.cos(angle)
    m2 = np.sin(angle)
    return np.array([
        m1 * vector[0] - m2 * vector[1],
        m1 * vector[1] + m2 * vector[0]
    ], np.float32)
    
def get_vector_angle(vector: Vec2):
    angle = np.arctan2(vector[1], vector[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle

def normalise_vector(vector: Vec2) -> Vec2:
    length = np.linalg.norm(vector)
    if length != 0:
        return np.array([vector[0] / length, vector[1] / length], np.float32)
    else:
        return np.array([0, 1], np.float32)

def get_perpendicular_vector(vector: Vec2) -> Vec2:
    return np.array([-vector[1], vector[0]], np.float32)

def length_angle_to_vector(length: float, angle: float) -> Vec2:
    return np.array([
        length * np.cos(angle),
        length * np.sin(angle)
    ], np.float32)
    
def get_reciprocal(vector: Vec2):
    return np.array([-vector[0], -vector[1]], np.float32)
###

DRAWABLE_RADIUS: float = 50.0

# BeamSensor Settings
class BeamSettings:
    SENSOR_ALPHA: float = 0.2
    BEAM_QUALITY: float = 0.1

# Agent Settings
class AgentSettings:
    RADIUS: float = 5.0
    MAX_SPEED: float = 100.0
    MIN_SPEED: float = -50.0
    MAX_ROTATE: float = 2 * np.pi
    DRAG: float = 50.0
    ACCELERATION: float = 5000.0
    TIMESTEP: float = 0.05
    PARTS: int = 4

class AgentPart:
    BODY = 0
    CENTRE = 1
    ARROW = 2
    WHEEL = 3

AGENT_COLOURS = np.zeros((4, 4))
AGENT_COLOURS[AgentPart.CENTRE][:] = 1.0
AGENT_COLOURS[AgentPart.ARROW][:3] = 0.0
AGENT_COLOURS[AgentPart.ARROW][3] = 1.0
AGENT_COLOURS[AgentPart.WHEEL][:3] = 0.1
AGENT_COLOURS[AgentPart.WHEEL][3] = 1.0

# Colour Functions
class ColourType:
    BLACK = 0
    WHITE = 1
    GREEN = 2
    BLUE = 3
    RED = 4
    PURPLE = 5
    DARK_PURPLE = 6
    YELLOW = 7
    LILAC = 8
    BROWN = 9
    LIGHT_GREY = 10
    DARK_GREY = 11
    MID_GREY = 12
    ORANGE = 13
    PINK = 14
    SELECTION = 15
    
ColourPalette = [
    [0.0, 0.0, 0.0, 1.0],  # black
    [1.0, 1.0, 1.0, 1.0],  # white
    [0.2, 0.8, 0.2, 1.0],  # green
    [0.2, 0.2, 0.8, 1.0],  # blue
    [0.8, 0.2, 0.2, 1.0],  # red
    [0.5, 0.3, 0.7, 1.0],  # purple
    [0.2, 0.0, 0.4, 1.0],  # dark purple
    [0.8, 0.8, 0.2, 1.0],  # yellow
    [0.8, 0.5, 0.9, 1.0],  # lilac
    [0.4, 0.3, 0.1, 1.0],  # brown
    [0.8, 0.8, 0.8, 1.0],  # light grey
    [0.3, 0.3, 0.3, 1.0],  # dark grey
    [0.5, 0.5, 0.5, 1.0],  # mid grey
    [0.9, 0.9, 0.1, 1.0],  # orange
    [1.0, 0.8, 0.8, 1.0],  # np.pink
    [0.5, 0.5, 1.0, 0.5]  # selected
]

def random_colour() -> list[float]:
    return np.random.rand(4).tolist()

MAX_COLLISIONS = 200

WORLD_DISPLAY_TYPE = SimpleNamespace(**{
    "DISPLAY_NONE": 0, # Nothing
    "DISPLAY_NONE": 1, # Agents
    "DISPLAY_WORLDOBJECTS": 2, # World Objects
    "DISPLAY_AGENTS": 2, # World Objects
    "DISPLAY_TRAILS": 4, # Trails
    "DISPLAY_SENSORS": 8, # Sensors
    "DISPLAY_COLLISIONS": 0, # Set to 16 to show collision
    "DISPLAY_MONITOR": 32, # Monitor
    "DISPLAY_ALL": 65535 # Everything
})

WORLD_DISPLAY_PARAMETERS = SimpleNamespace(**{
    "width": 800.0,
    "height": 600.0,
    "window_width": 800.0,
    "window_height": 600.0,
    "config": 65535,
    "colour": [1.0, 1.0, 1.0],
    "dimension": 0
})

BACKGROUND_COLOUR = ColourType.DARK_PURPLE

FFN_ACTIVATION_RESPONSE: float = 0.5

GA_SELECTION_TYPE = SimpleNamespace(
    ROULETTE = 0,
    RANK = 1,
    TOURNAMENT = 2
)

GA_PRINT_TYPE = SimpleNamespace(
    PARAMETERS = 1,
    CURRENT = 2,
    GENERATION = 4,
    HISTORY = 8
)

GA_FITNESS_METHOD = SimpleNamespace(
    BEST = 0,
    WORST = 1,
    MEAN = 2,
    TOTAL = 3
)

GA_FITNESS_FIX = SimpleNamespace(
    IGNORE = 0,
    CLAMP = 1,
    FIX = 2
)

GA_FLOAT_DEFAULT = SimpleNamespace(
    TOURNAMENT = 0.75,
    RANK_SPRESSURE = 1.5,
    EXPONENT = 1.0
)

GA_INT_DEFAULT = SimpleNamespace(
    TOURNAMENT_SIZE = 5
)