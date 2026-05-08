from core.agent.agent import Agent

class NeuralAgent(Agent):
    def __init__(self):
        super().__init__()
        self.brain = None
        self._has_brain: bool = False
    
    def __del__(self):
        if self.brain is not None:
            del self.brain
            self._has_brain = False