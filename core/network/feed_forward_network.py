import numpy as np

from core.utils import FFN_ACTIVATION_RESPONSE

class Neuron:
    def __init__(self, size: int, bias: bool):
        self.weights = np.zeros(size + (1 if bias else 0))
        self._bias = bias
    
    def weighted_sum(self, values):
        if self._bias:
            return np.dot(self.weights[:-1], values) + self.weights[-1]
        else:
            return np.dot(self.weights, values)

class FeedForwardNetwork:
    def __init__(
        self,
        inputs: int,
        outputs: int,
        hidden_nodes: int = 0,
        sigmoid: bool = True,
        bias: bool = True
    ):
        self._inputs = inputs
        self._outputs = outputs
        self._hidden_nodes = hidden_nodes
        self._sigmoid = sigmoid
        self._bias = bias
        
        self.input_values = np.zeros(self._inputs, dtype=np.float32)
        self.output_values = np.zeros(self._outputs, dtype=np.float32)
        self._input_to_hidden: int = 0
        self._hidden_to_output: int = 0
        self.number_weights: int = 0
        
        self._hidden_layer = []
        self._output_layer = []
    
        self.initialise(inputs, outputs, hidden_nodes, sigmoid, bias)
        
    def initialise(
        self,
        inputs: int,
        outputs: int,
        hidden_nodes: int,
        sigmoid: bool = True,
        bias: bool = True
    ) -> None:
        # NOTE: Yes this is duplication, but allows for better intellisense
        self._inputs = inputs
        self._outputs = outputs
        self._hiddden_nodes = hidden_nodes
        self._sigmoid = sigmoid
        self._bias = bias
        
        self.input_values = np.zeros(self._inputs, dtype=np.float32)
        self.output_values = np.zeros(self._outputs, dtype=np.float32)
        self._input_to_hidden: int = (self._inputs + (1 if self._bias else 0)) * self._hiddden_nodes
        self._hidden_to_output: int = (self._hidden_nodes + (1 if self._bias else 0)) * self._outputs
        self.number_weights: int = self._input_to_hidden + self._hidden_to_output
        
        self._hidden_layer: list[Neuron] = []
        self._output_layer: list[Neuron] = []
        
        for _ in range(self._hiddden_nodes):
            self._hidden_layer.append(Neuron(self._inputs, self._bias))
        
        # If no hidden nodes, map input layer to output layer
        if self._hidden_nodes == 0:
            self._hiddden_nodes = self._inputs
        
        for _ in range(outputs):
            self._output_layer.append(Neuron(self._hiddden_nodes, self._bias))
    
    def fire(self) -> None:
        hidden_output = []
        output_values = []
        
        if self._hiddden_nodes == 0:
            hidden_output = self.input_values
        
        for neuron in self._hidden_layer:
            output = neuron.weighted_sum(self.input_values)
            output = self.activation_function(output)
            hidden_output.append(output)
        
        for neuron in self._output_layer:
            output = neuron.weighted_sum(hidden_output)
            output = self.activation_function(output)
            output_values.append(output)
        
        self.output_values = output_values
    
    def randomise(self) -> None:
        for neuron in self._hidden_layer:
            neuron.weights = np.random.uniform(-1.0, 1.0, len(neuron.weights))
        for neuron in self._output_layer:
            neuron.weights = np.random.uniform(-1.0, 1.0, len(neuron.weights))
    
    def activation_function(self, x: float) -> float:
        if self._sigmoid:
            return 2.0 / (1.0 + np.exp(-x / FFN_ACTIVATION_RESPONSE)) - 1.0
        else:
            return 1.0 if x > 0.0 else 0.0
    
    def set_activation_function(self, function: callable):
        setattr(self, "activation_function", function)
    
    def set_configuration(self, config: dict):
        assert len(config["hidden"]) == self._hiddden_nodes, "Inconsistent number of hidden neurons"
        assert len(config["output"]) == self._outputs, "Inconsistent number of output neurons"
        for neuron, weights in zip(self._hidden_layer, config["hidden"]):
            assert len(weights) == len(neuron.weights), "Number of hidden neuron weights does not equal number of inputs"
            neuron.weights[:] = weights
        for neuron, weights in zip(self._output_layer, config["output"]):
            assert len(weights) == len(neuron.weights), "Number of output neuron weights does not equal number of hidden neurons"
            neuron.weights[:] = weights
    
    def get_configuration(self) -> dict:
        return {
            "hidden": [ neuron.weights for neuron in self._hidden_layer ],
            "output": [ neuron.weights for neuron in self._output_layer ]
        }

    # TODO: Set/Get Configuration?
    # TODO: Serialise/deserialise?