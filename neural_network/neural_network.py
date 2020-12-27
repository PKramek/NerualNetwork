import numpy as np

from .layers import Layer
from .loss_functions import LossFunction


class NeuralNetwork:
    def __init__(self, loss_function: 'LossFunction'):
        self.layers = []
        self.loss_function = loss_function

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: 'LossFunction'):
        assert isinstance(loss_function, LossFunction), 'Loss function must be object of class extending LossFunction'
        self._loss_function = loss_function

    def add(self, layer: 'Layer'):
        self.layers.append(layer)

    def predict(self, input_data: np.ndarray):
        # input data could be 2d
        output = self.forward_propagation(input_data)
        return output.flatten()

    def forward_propagation(self, input_data: np.array):
        output = input_data

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward_propagation(self, error: float, learning_rate: float):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

        return error

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float):
        assert isinstance(epochs, int) and epochs > 0, 'Number of epochs must be integer bigger than 0'
        assert isinstance(learning_rate, float) and learning_rate > 0, 'Learning rate must be float bigger than 0'

        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)

        errors = []

        # TODO test if indices are correct
        assert x_train.shape[0] == y_train.shape[0]

        num_of_samples = len(y_train)

        for epoch in range(epochs):
            epoch_error = 0
            for i in range(num_of_samples):
                output = self.forward_propagation(x_train[i])

                error = self._loss_function.loss(output, y_train[i])
                # TODO change this to use mean from epoch not all errors
                epoch_error += error

                error_derivative = self._loss_function.derivative(output, y_train[i])
                self.backward_propagation(error_derivative, learning_rate)

            errors.append(epoch_error)

        return errors

    def score(self, x_true: np.array, y_true: np.array):
        predictions = self.predict(x_true)
        diff = y_true - predictions
        u = np.sum(diff ** 2)
        true_mean = np.mean(y_true)
        second_diff = (y_true - true_mean)
        v = np.sum(second_diff ** 2)
        score = 1 - (u / v)

        return score
