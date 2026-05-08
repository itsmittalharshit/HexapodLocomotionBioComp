from OpenGL.GL import *
from OpenGL.GLU import *
from core.world.drawable import Drawable
from core.utils import Vec2, MAX_COLLISIONS

class Collision(Drawable):
    def __init__(self, location: Vec2, visible: bool = False):
        super().__init__(location, visible=visible)
    
    def display(self) -> None:
        if not self.visible:
            return
        disk = gluNewQuadric()
        gluQuadricDrawStyle(disk, GLU_FILL)
        glColor4f(0.9, 0.9, 0.4, 0.2)
        
        glEnable(GL_BLEND)
        glPushMatrix()
        glTranslated(self.location[0], self.location[1], 0)
        gluDisk(disk, 0, 3, 10, 1)
        glPopMatrix()
        glDisable(GL_BLEND)
        
        gluDeleteQuadric(disk)

class Collisions:
    def __init__(self):
        self.collisions: list[Collision] = []
    
    def __len__(self):
        return len(self.collisions)
    
    def append(self, c: Collision) -> None:
        self.collisions.append(c)
    
    def update(self) -> None:
        self.collisions = self.collisions[:MAX_COLLISIONS]
    
    def clear(self) -> None:
        self.collisions.clear()
    
    def display(self) -> None:
        for c in self.collisions:
            c.display()
