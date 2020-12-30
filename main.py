import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

from neural_network.layers import Linear, Sigmoid, ReLu, Tanh
from neural_network.loss_functions import MSE
from neural_network.neural_network import NeuralNetwork

np.random.seed(42)

warnings.filterwarnings("ignore")


def aim_function(x):
    return x ** 3


def create_samples(func, num):
    x_s = np.random.uniform(low=-2, high=2, size=(num, 1))
    y_s = np.array([np.array([func(i[0])]) for i in x_s], dtype=np.float64)
    return x_s, y_s


def create_training_and_testing_data(training_size: int, test_size: int):
    x_train, y_train = create_samples(aim_function, training_size)
    x_test, y_test = create_samples(aim_function, test_size)

    return x_train, y_train, x_test, y_test


def create_network(activation_type):
    activation_lookup = {
        'Tanh': Tanh,
        'Sigmoid': Sigmoid,
        'ReLu': ReLu
    }
    activation = activation_lookup.get(activation_type, None)
    if activation is None:
        raise ValueError('Not know activation function')

    nn = NeuralNetwork(MSE())

    first = Linear(1, 100)
    first_activation = activation(100, 100)
    second = Linear(100, 100)
    second_activation = activation(100, 100)
    third = Linear(100, 1)

    nn.add(first)
    nn.add(first_activation)
    nn.add(second)
    nn.add(second_activation)
    nn.add(third)

    return nn


def test_network_and_default_implementation(
        activation_type, epochs: int = 1000, learning_rate=0.001, minibatch_size=64, n: int = 10,
        train_size: int = 1000, test_size=100):
    assert isinstance(n, int) and n > 0
    nn_scores = []
    regr_scores = []

    for i in range(n):
        x_train, y_train, x_test, y_test = create_training_and_testing_data(train_size, test_size)

        nn = create_network(activation_type)
        nn.train(x_train.T, y_train.T, learning_rate=learning_rate, epochs=epochs, minibatch_size=minibatch_size)
        nn_scores.append(nn.score(x_test.T, y_test.T))

        regr = MLPRegressor().fit(x_train, y_train.ravel())
        regr_scores.append(regr.score(x_test, y_test.ravel()))

    return nn_scores, np.mean(nn_scores), np.std(nn_scores), regr_scores, np.mean(regr_scores), np.std(regr_scores)


def plot_errors(path, errors, test_errors, title):
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value")
    plt.title(title)
    plt.plot(errors, label="Train data")
    plt.plot(test_errors, label="Test data")
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.show()


learning_rate = 0.001
minibatch_size = 64
epochs = 200
train_size = 1000
test_size = 100
n = 10  # number of experiment repetitions

x_train, y_train, x_test, y_test = create_training_and_testing_data(train_size, test_size)

activation_functions = ['Tanh', 'Sigmoid', 'ReLu']
results_dict = {}

for activation in activation_functions:
    nn = create_network(activation)
    errors, test_errors = nn.train(
        x_train.T, y_train.T, learning_rate=learning_rate, epochs=epochs, minibatch_size=minibatch_size,
        calc_test_err=True, x_test=x_test.T, y_test=y_test.T)

    plot_errors(f'results/{activation}.png', errors, test_errors, activation)

    print(f'{activation} score: {nn.score(x_test.T, y_test.T)}')

    results = test_network_and_default_implementation(
        activation, epochs, learning_rate, minibatch_size, n, train_size, test_size)
    results_dict[activation] = {
        'nn scores': results[0],
        'nn mean score': results[1],
        'nn score std': results[2],
        'regr scores': results[3],
        'regr mean score': results[4],
        'regr score std': results[5],
    }

pprint(results_dict)
