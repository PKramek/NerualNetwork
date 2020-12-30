from pprint import pprint
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from neural_network.layers import Linear, Sigmoid, ReLu, Tanh
from neural_network.loss_functions import MSE
from neural_network.neural_network import NeuralNetwork

np.random.seed(42)


# warnings.filterwarnings("ignore")


def aim_function(x):
    return x ** 4 / 100


def create_samples(func: Callable, num: int):
    x_s = np.random.uniform(low=-2, high=2, size=(num, 1))
    y_s = np.array([np.array([func(i[0])]) for i in x_s], dtype=np.float64)
    return x_s, y_s


def create_training_and_testing_data(n_samples: int, test_size: float, n_features: int):
    assert isinstance(test_size, float) and 0 < test_size < 1
    assert isinstance(n_samples, int) and n_samples > 0
    assert isinstance(n_features, int) and n_features > 0

    x, y = make_regression(n_samples=n_samples, n_features=n_features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])

    return x_train, y_train, x_test, y_test


def create_network(activation_type: str, input_size: int, output_size: int, hidden_layer_size: int = 100):
    assert isinstance(input_size, int) and input_size > 0
    assert isinstance(output_size, int) and output_size > 0
    assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0

    activation_lookup = {
        'Tanh': Tanh,
        'Sigmoid': Sigmoid,
        'ReLu': ReLu
    }
    activation = activation_lookup.get(activation_type, None)
    if activation is None:
        raise ValueError('Not know activation function')

    nn = NeuralNetwork(MSE())

    first = Linear(input_size, hidden_layer_size)
    first_activation = activation(hidden_layer_size, hidden_layer_size)
    second = Linear(hidden_layer_size, hidden_layer_size)
    second_activation = activation(hidden_layer_size, hidden_layer_size)
    third = Linear(hidden_layer_size, output_size)

    nn.add(first)
    nn.add(first_activation)
    nn.add(second)
    nn.add(second_activation)
    nn.add(third)

    return nn


def test_network(x_train, y_train, x_test, y_test, activation_type, num_epochs: int = 200,
                 lr=0.001, minibatch_size: int = 32, n: int = 10):
    assert isinstance(n, int) and n > 0
    nn_scores = []

    for i in range(n):
        nn = create_network(activation_type, n_features, output_size=1)
        nn.train(x_train.T, y_train, learning_rate=lr, epochs=num_epochs, minibatch_size=minibatch_size)
        nn_scores.append(nn.score(x_test.T, y_test))

    return nn_scores, np.mean(nn_scores), np.std(nn_scores)


def test_default_implementation(x_train, y_train, x_test, y_test, n: int = 10):
    assert isinstance(n, int) and n > 0
    regr_scores = []

    for i in range(n):
        regr = MLPRegressor().fit(x_train, y_train.ravel())
        regr_scores.append(regr.score(x_test, y_test.ravel()))

    return regr_scores, np.mean(regr_scores), np.std(regr_scores)


def plot_errors(path: str, errors: List[float], test_errors: List[float], title: str):
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
minibatch_size = 32
epochs = 200
n_samples = 1000
test_size = 0.2
n_features = 10
n = 10  # number of experiment repetitions

x_train, y_train, x_test, y_test = create_training_and_testing_data(n_samples, test_size, n_features)

activation_functions = ['ReLu', 'Tanh', 'Sigmoid']
results_dict = {}

for activation in activation_functions:
    neural_network = create_network(activation, n_features, output_size=1)
    errors, test_errors = neural_network.train(
        x_train.T, y_train, learning_rate=learning_rate, epochs=epochs, minibatch_size=minibatch_size,
        calc_test_err=True, x_test=x_test.T, y_test=y_test)

    plot_errors(f'results/{activation}.png', errors, test_errors, activation)

    print(f'{activation} score: {neural_network.score(x_test.T, y_test)}')

    results = test_network(
        x_train, y_train, x_test, y_test, activation, epochs, learning_rate, minibatch_size, n)
    results_dict[activation] = {
        'scores': results[0],
        'mean score': results[1],
        'score std': results[2]
    }

results = test_default_implementation(x_train, y_train, x_test, y_test, n)
results_dict['MLPRegressor'] = {
    'scores': results[0],
    'mean score': results[1],
    'score std': results[2]
}

pprint(results_dict)
