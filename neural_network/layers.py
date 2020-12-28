from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, input_size: int, output_size: int):
        assert isinstance(input_size, int) and input_size > 0, 'Input size must me integer bigger than 0'
        assert isinstance(output_size, int) and input_size > 0, 'Output size must me integer bigger than 0'

        self._input_size = input_size
        self._output_size = output_size

        # Used in backprop
        self.input = None
        self.output = None

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @abstractmethod
    def forward(self, input_data: np.array):
        pass

    @abstractmethod
    def backward(self, error: np.array, learning_rate: float):
        pass


# TODO test these classes

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

        weights_low = -1 / np.sqrt(input_size)
        weights_high = 1 / np.sqrt(input_size)

        self.weights = np.random.uniform(low=weights_low, high=weights_high, size=(output_size, input_size))
        self.bias = np.zeros((output_size, 1), dtype=np.float32)

    def forward(self, input_data: np.ndarray):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias

        return self.output

    def backward(self, grad: np.array, learning_rate: float):
        input_derivatives = np.dot(self.weights.T, grad)
        weights_derivatives = np.dot(grad, self.input.T)
        bias_derivatives = np.sum(grad, axis=1, keepdims=True)

        self.weights -= learning_rate * weights_derivatives
        self.bias -= learning_rate * bias_derivatives

        return input_derivatives


class ReLu(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, input_data: np.array):
        self.input = input_data
        self.output = input_data * (input_data > 0)

        return self.output

    def backward(self, error: np.array, learning_rate: float):
        return (self.input > 0) * error


class Tanh(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, input_data: np.array):
        self.input = input_data
        self.output = np.tanh(input_data)

        return self.output

    def backward(self, error: np.array, learning_rate: float):
        return (1 - np.tanh(self.input) ** 2) * error


class Sigmoid(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, input_data: np.array):
        self.input = input_data
        self.output = np.tanh(input_data)

        return self.output

    def backward(self, error: np.array, learning_rate: float):
        return error * self.input * (1 - self.input)
