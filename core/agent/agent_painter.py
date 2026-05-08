from core.agent.agent import Agent

MONITOR_BAR_HEIGHT = 25

class AgentPainter():
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.visible: bool = True
