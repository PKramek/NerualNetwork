from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, input_size: int, output_size: int):
        assert isinstance(input_size, int) and input_size > 0, 'Input size must me integer bigger than 0'
        assert isinstance(output_size, int) and input_size > 0, 'Output size must me integer bigger than 0'

        self._input_size = input_size
        self._output_size = output_size

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


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

        weights_low = -1 / np.sqrt(input_size)
        weights_high = 1 / np.sqrt(input_size)

        self.weights = np.random.uniform(low=weights_low, high=weights_high, size=(input_size, output_size))
        self.bias = np.random.rand(1, output_size)

        self.input = None
        self.output = None

    def forward(self, input_data: np.array):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward(self, error: np.array, learning_rate: float):
        pass
