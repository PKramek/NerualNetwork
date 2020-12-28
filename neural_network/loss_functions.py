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
        m = output.shape[1]

        cost = np.sum(np.square(desired_output - output)) / m
        return np.squeeze(cost)

    def derivative(self, output: np.array, desired_output: np.array):
        m = output.shape[1]
        return (2 * (output - desired_output) / len(output)) / m
