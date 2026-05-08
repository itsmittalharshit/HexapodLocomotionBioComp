import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from core.world.collisions import Collisions, Collision
from core.utils import Vec2, Vec3, WORLD_DISPLAY_PARAMETERS, WORLD_DISPLAY_TYPE, ColourPalette, BACKGROUND_COLOUR
from core.agent.agent import Agent
from core.world.world_object import WorldObject

class World:
    def __init__(self, simulation):
        self._simulation = simulation
        self._agents: list[Agent] = []
        self._agent_queue: list[Agent] = []
        self._objects: list[WorldObject] = []
        self._object_queue: list[WorldObject] = []
        self._collisions = Collisions()
        
        self._colour = None
        self._update_in_progress: bool = False
        
        self._display_type = WORLD_DISPLAY_TYPE
        self._display_params = WORLD_DISPLAY_PARAMETERS
        
        # TODO: Mouse compatibility? self.mouse
        # TODO: Keyboard compatibilit? self.keys
        
        self.eye: Vec3 = np.array([
            0.5 * self._display_params.width,
            self._display_params.height,
            100.0
        ], np.float32)
        self.look: Vec3 = np.array([
            0.5 * self._display_params.width,
            0.5 * self._display_params.height,
            0.0
        ], np.float32)
        self.up: Vec3 = np.array([0.0, 0.0, 1.0], np.float32)
        
    def initialise(self) -> None:
        for obj in self._objects:
            obj.initialise()
        for agt in self._agents:
            agt.initialise()
    
    def _initialise_gl(self) -> None:
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glEnable(GL_COLOR_MATERIAL)
        
        global_ambient = [0.3, 0.3, 0.3, 1.0]
        diffuse = [1.0, 1.0, 1.0, 1.0]
        specular = [1.0, 1.0, 1.0, 1.0]
        
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient)
        position = [
            0.0, 
            0.5 * self._display_params.height, 
            0.5 * self._display_params.width, 
            1.0
        ]
        glLightfv(GL_LIGHT0, GL_POSITION, position)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular)
        glEnable(GL_LIGHT0)
        
        glShadeModel(GL_SMOOTH)
        glClearColor(
            self._display_params.colour[0],
            self._display_params.colour[1],
            self._display_params.colour[2],
            1.0
        )
        glMatrixMode(GL_PROJECTION)
        gluOrtho2D(0, self._display_params.width, 0, self._display_params.height)
        glMatrixMode(GL_MODELVIEW)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    
    def add_object(self, obj: list[WorldObject | Agent] | WorldObject | Agent) -> None:
        if isinstance(obj, list):
            for o in obj:
                self.add_object(o)
        elif isinstance(obj, Agent):
            if not self._update_in_progress:
                self._agents.append(obj)
            else:
                self._agent_queue.append(obj)
        elif isinstance(obj, WorldObject):
            if not self._update_in_progress:
                self._objects.append(obj)
            else:
                self._object_queue.append(obj)
        obj.world = self
    
    def add_collision(self, vector: Vec2):
        c = Collision(vector, bool(self._display_type.DISPLAY_COLLISIONS))
        self._collisions.append(c)
    
    def remove_object(self, typing: type[WorldObject | Agent]) -> list[WorldObject | Agent]:
        if self._update_in_progress:
            return
        
        removed_objects = []
        for obj in reversed(self._objects[:]):
            if isinstance(obj, typing):
                self._objects.remove(obj)
                removed_objects.append(obj)
        
        removed_agents = []
        for agt in reversed(self._agents[:]):
            if isinstance(agt, typing):
                self._agents.remove(agt)
                self._agent_queue.append(agt)
        
        # TODO: Monitor update?
        return removed_objects + removed_agents
    
    def display(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self._colour = ColourPalette[BACKGROUND_COLOUR][:3]
        glClearColor(self._colour[0], self._colour[1], self._colour[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self._display_params.config & self._display_type.DISPLAY_WORLDOBJECTS != 0:
            for obj in self._objects:
                obj.display()
        if self._display_params.config & self._display_type.DISPLAY_AGENTS != 0:
            for agt in self._agents:
                agt.display()
        if self._display_params.config & self._display_type.DISPLAY_COLLISIONS != 0:
            self._collisions.display()
    
    def draw_objects(self) -> None:
        if self._display_params.config & self._display_type.DISPLAY_WORLDOBJECTS != 0:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            # TODO: does this work?
            gluLookAt(*self.eye, *self.look, *self.up)
            for i in range(1, len(self._objects) + 1):
                glLoadName(i)
                self._objects[i - 1].display()
            glFlush()
    
    def update(self) -> None:
        self._update_in_progress = True
        # TODO: Mouse update?
        
        for obj in self._objects:
            obj.update()
        for agt in self._agents:
            agt.update()
        
        for obj in reversed(self._objects[:]):
            if obj.dead:
                self._objects.remove(obj)
        for agt in reversed(self._agents[:]):
            if agt.dead:
                self._agents.remove(agt)
        
        if self._agents:
            for obj in self._objects:
                for agt in self._agents:
                    agt.interact(obj)
            for i, agt1 in enumerate(self._agents):
                for j, agt2 in enumerate(self._agents):
                    if i != j:
                        agt1.interact(agt2)
        
        self._collisions.update()
        self._update_in_progress = False
        self._update_queues()
    
    def _update_queues(self) -> None:
        self._agents.extend(self._agent_queue)
        self._agent_queue.clear()
        self._objects.extend(self._object_queue)
        self._object_queue.clear()
    
    def clean(self) -> None:
        self._agents.clear()
        self._objects.clear()
        self._collisions.clear()
    
    def centre(self) -> Vec2:
        return np.array([
            0.5 * self._display_params.width,
            0.5 * self._display_params.height
        ], np.float32)
    
    def random_location(self) -> Vec2:
        return np.array([
            self._display_params.width * np.random.rand(),
            self._display_params.height * np.random.rand()
        ], np.float32)
    
    def to_window_coords(self, x: float, y: float) -> Vec2:
        return np.array([
            x / self._display_params.window_width * self._display_params.width,
            ((self._display_params.window_height - y) / self._display_params.window_height) * self._display_params.height
        ], np.float32)
    
