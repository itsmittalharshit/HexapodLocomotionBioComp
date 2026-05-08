import numpy as np

from core.world.drawable import Drawable
from core.utils import Vec2, DRAWABLE_RADIUS, get_vector_angle, normalise_vector, get_perpendicular_vector

class WorldObject(Drawable):
    _amount = 0 # Amount of World Objects
    
    def __init__(
        self,
        location: Vec2 = None,
        orientation: float = None,
        radius: float = DRAWABLE_RADIUS,
        edges: list[Vec2] = None,
        solid: bool = False
    ):
        super().__init__(
            location,
            orientation,
            radius,
            edges = edges
        )
        
        self._absolute_edges = []
        self._reset_random = {}
        
        self._reset_random["location"] = True if location is None else False
        self._reset_random["orientation"] = True if orientation is None else False
        
        self.solid = solid
        self.initialised = False
        WorldObject._amount += 1
    
    def __del__(self):
        WorldObject._amount -= 1
        super().__del__()
    
    def initialise(self) -> None:
        if self._start_location is None:
            if self.world is not None:
                self._start_location = self.world.random_location()
            else:
                self._start_location = np.array([0.0, 0.0], np.float32)
        if self._start_orientation is None:
            self._start_orientation = np.random.uniform(high=2*np.pi)
        
        self.location = self._start_location
        self.orientation = self._start_orientation
        
        if not self.circular:
            self.calc_absolute_edges()
        
        self.dead = False
        super().initialise()
        self.initialised = True
    
    def reset(self) -> None:
        if self._reset_random["location"]:
            # TODO: Random Location function in World
            self._start_location = self.world.random_location()
        if self._reset_random["orientation"]:
            self._start_orientation = np.random.uniform(high=2*np.pi)
        self.location = self._start_location
        self.orientation = self._start_orientation
    
    # TODO: Undefined in Pybeast
    def interact(self, other: "WorldObject"):
        pass
    
    # NOTE: Not implemented here, but in Agent
    def update(self):
        pass
    
    def on_collision(self, other) -> None:
        pass

    def is_inside(self, vector: Vec2) -> bool:
        if self.circular:
            return np.linalg.norm(vector - self.location)**2 <= self.radius**2
        
        vec_out = vector + np.array([self.radius + 1.0, 0], np.float32)
        inside = False
        for edge, next_edge in zip(self._absolute_edges[:-1], self._absolute_edges[1:]):
            if (edge[1] >= vector[1] and next_edge[1] < vector[1]) or (edge[1] < vector[1] and next_edge[1] >= v[1]):
                if self.intersect(edge, next_edge, vector, vec_out) is not None:
                    inside = not inside
        return inside
    
    def calc_absolute_edges(self):
        self._absolute_edges.clear()
        m1 = np.cos(self.orientation)
        m2 = np.sin(self.orientation)
        
        for edge in self.edges:
            self._absolute_edges.append(np.array([
                self.location[0] + (m1 * edge[0] - m2 * edge[1]),
                self.location[1] + (m1 * edge[1] + m2 * edge[0])
            ], np.float32))
        
    def intersect(self, a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2):
        delta_a_gradient = np.gradient(a2 - a1)
        delta_b_gradient = np.gradient(b2 - b1)
        delta_a_intersect = a1[1] - delta_a_gradient * a1[0]
        delta_b_intersect = b1[1] - delta_b_gradient * b1[0]
        
        # If parallel
        if delta_a_gradient == delta_b_gradient:
            return None
        
        if delta_a_gradient == np.inf:
            r = np.array([a1[0], delta_b_gradient * a1[0] + delta_b_intersect], np.float32)
            if (((r[1] <= a1[1] and r[1] >= a2[1])) or (r[1] <= a2[1] and r[1] >= a1[1])
                and ((r[1] <= b1[1] and r[1] >= b2[1]) or (r[1] <= b2[1] and r[1] >= b1[1]))
                and ((r[0] <= b1[0] and r[0] >= b2[0]) or (r[0] <= b2[0] and r[0] >= b1[0]))):
                return r
        
        if delta_b_gradient == np.inf:
            r = np.array([b1[0], delta_a_gradient * b1[0] + delta_a_intersect], np.float32)
            if (((r[1] <= a1[1] and r[1] >= a2[1])) or (r[1] <= a2[1] and r[1] >= a1[1])
                and ((r[1] <= b1[1] and r[1] >= b2[1]) or (r[1] <= b2[1] and r[1] >= b1[1]))
                and ((r[0] <= a1[0] and r[0] >= a2[0]) or (r[0] <= a2[0] and r[0] >= a1[0]))):
                return r
        
        r = np.array([-(delta_a_intersect - delta_b_intersect) / (delta_a_gradient - delta_b_gradient), 0.0], np.float32)
        r[1] = delta_a_gradient * r[0] + delta_a_intersect
        
        if (((r[0] <= a1[0] and r[0] >= a2[0]) or (r[0] <= a2[0] and r.x >= a1[0]))
            and ((r[0] <= b1[0] and r[0] >= b2[0]) or (r[0] <= b2[0] and r[0] >= b1[0]))):
            return r

        return None

    def _nearest_point_on_line(self, vector: Vec2, l1: Vec2, l2: Vec2) -> Vec2:
        A = vector - l1
        B = l2 - l1
        theta = np.pi / 2 - (get_vector_angle(B) - get_vector_angle(A))
        distance_alongside = np.linalg.norm(A) * np.sin(theta)
        
        if distance_alongside > np.linalg.norm(B):
            return l2
        elif distance_alongside <= 0:
            return l1
        else:
            return l1 + normalise_vector(B) * distance_alongside
    
    def nearest_point(self, vector: Vec2):
        if self.circular:
            collision_normal = normalise_vector(vector - self.location)
            collision_point = self.location + collision_normal * self.radius
            return collision_point, collision_normal
        
        side_found = False
        v1, v2 = None, None
        # TODO: Check if self._absolute_edges are vertices or edges?
        for vertex, next_vertex in zip(self._absolute_edges[:-1], self._absolute_edges[1:]):
            vector = self.intersect(vertex, next_vertex, vector, self.location)
            if vector:
                v1, v2 = vertex, next_vertex
                side_found = True
                break
        
        if not side_found:
            return vector
        
        # NOTE: Surely doesn't ever run or breaks?
        collision_point = self._nearest_point_on_line(vector, v1, v2)
        collision_normal = normalise_vector(get_perpendicular_vector(v2 - v1))
        return collision_point, collision_normal