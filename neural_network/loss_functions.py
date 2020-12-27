from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, output: np.array, desired_output: np.array):
        pass

    @abstractmethod
    def derivative(self, output: np.array, desired_output: np.array):
        pass


class MSE(LossFunction):
    def loss(self, output: np.array, desired_output: np.array):
        return np.mean(np.power(desired_output - output, 2))

    def derivative(self, output: np.array, desired_output: np.array):
        return 2 * (output - desired_output) / len(output)
