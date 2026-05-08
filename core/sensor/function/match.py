from core.sensor.base import MatchFunction
from core.world.world_object import WorldObject

class MatchKind(MatchFunction):
    """
    Matches if instance of class or class that inherits from class
    """
    def __init__(self, object_type: type[WorldObject]):
        self.object_type = object_type
    
    def __call__(self, obj: WorldObject) -> bool:
        return isinstance(obj, self.object_type) or issubclass(type(obj), self.object_type)

class MatchExact(MatchFunction):
    """
    Matches if instance of class
    """
    def __init__(self, object_type: type[WorldObject] = WorldObject):
        self.object_type = object_type
    
    def __call__(self, obj: WorldObject) -> bool:
        return isinstance(obj, self.object_type)

class MatchInstance(MatchFunction):
    """
    Matches an instance
    """
    def __init__(self, obj: WorldObject):
        self.target = obj
    
    def __call__(self, obj: WorldObject) -> bool:
        return obj is self.target

class MatchComposeOr:
    """
    Logical OR between two MatchFunction instances
    """
    def __init__(
        self,
        first: MatchFunction = None,
        second: MatchFunction = None
    ):
        self.functions = [first, second]
        
    def __del__(self):
        for f in self.functions:
            del f
    
    def __call__(self, obj: WorldObject) -> bool:
        for f in self.functions:
            if f(obj):
                return True
        return False
    
class MatchComposeAnd:
    """
    Logical AND between two MatchFunction instances
    """
    def __init__(
        self,
        first: MatchFunction = None,
        second: MatchFunction = None
    ):
        self.functions = [first, second]
        
    def __del__(self):
        for f in self.functions:
            del f
    
    def __call__(self, obj: WorldObject) -> bool:
        for f in self.functions:
            if not f(obj):
                return False
        return True

class MatchAdapter(MatchFunction):
    """
    Allows any unary function to be a MatchFunction
    """
    def __init__(self, function):
        self.function = function
    
    def __call__(self, obj: WorldObject) -> bool:
        return self.function(obj)