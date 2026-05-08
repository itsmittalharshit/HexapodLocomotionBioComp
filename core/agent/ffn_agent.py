import numpy as np

from core.agent.neural_agent import NeuralAgent
from core.network.feed_forward_network import FeedForwardNetwork
from core.evolve.evolver import Evolver

class FFNAgent(NeuralAgent):
    def __init__(self):
        super().__init__()
        
    def add_brain(
        self,
        hidden_nodes: int = -1,
        inputs: int = -1,
        outputs: int = -1,
        bias: bool = True
    ) -> None:
        if hidden_nodes == -1:
            hidden_nodes = len(self.sensors)
        if inputs == -1:
            inputs = len(self.sensors)
        if outputs == -1:
            outputs = len(self.controls)
        
        self.brain = FeedForwardNetwork(inputs, outputs, hidden_nodes, bias=bias)
        self.brain.randomise()
        self._has_brain = True
    
    def brain_output(self):
        for i, sensor in enumerate(self.sensors.values()):
            self.brain.input_values[i] = sensor.output()
            i += 1
        self.brain.fire()
        return self.brain.output_values
    
    def control(self):
        outputs = self.brain_output()
        for control, output in zip(self.controls.keys(), outputs):
            self.controls[control] = output

    # TODO: Serialise/deserialise?

class EvolvableFFNAgent(FFNAgent, Evolver):
    def __init__(self):
        FFNAgent.__init__(self)
        Evolver.__init__(self)
    
    def set_genotype(self, genome: list[float]):
        assert len(genome) == self.brain.number_weights, "Genome parameters must equal network weight quantity"
        input_to_hidden = genome[:self.brain._input_to_hidden].reshape(
            (self.brain._hiddden_nodes, self.brain._inputs + (1 if self.brain._bias else 0))
        )
        hidden_to_output = genome[self.brain._input_to_hidden:].reshape(
            (self.brain._outputs, self.brain._hiddden_nodes + (1 if self.brain._bias else 0))
        )
        self.brain.set_configuration({
            "hidden": input_to_hidden,
            "output": hidden_to_output
        })

    def get_genotype(self) -> list[float]:
        config = self.brain.get_configuration()
        return np.concatenate((
            np.array(config["hidden"]).flatten(),
            np.array(config["output"]).flatten()
        ))