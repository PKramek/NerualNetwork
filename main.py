import numpy as np

from neural_network.layers import Linear, ReLu
from neural_network.loss_functions import MSE
from neural_network.neural_network import NeuralNetwork

np.random.seed(42)
learning_rate = 0.01
epochs = 10000

first = Linear(5, 3)
relu = ReLu(3, 3)
second = Linear(3, 2)
second_relu = ReLu(2, 2)

input = np.ones(5, dtype=np.float32)

nn = NeuralNetwork(MSE())

nn.add(first)
nn.add(relu)
nn.add(second)
nn.add(second_relu)

nn.train()
