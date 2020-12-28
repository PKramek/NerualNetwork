import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

from neural_network.layers import Linear, ReLu
from neural_network.loss_functions import MSE
from neural_network.neural_network import NeuralNetwork


def aim_function(x):
    return (x ** 2) / 10


def create_samples(func, num):
    x_s = np.random.uniform(low=-10, high=10, size=(num, 1))
    y_s = np.array([np.array([func(i[0])]) for i in x_s], dtype=np.float32)
    return x_s, y_s


# np.random.seed(42)

x_samples, y_samples = create_samples(aim_function, 1000)

test_x, test_y = create_samples(aim_function, 10)

learning_rate = 0.001
epochs = 500

first = Linear(1, 100)
relu = ReLu(100, 100)
second = Linear(100, 1)

nn = NeuralNetwork(MSE())

nn.add(first)
nn.add(relu)
nn.add(second)


print('*' * 50 + 'Ours' + '*' * 50)
errors = nn.train(x_samples.T, y_samples.T, learning_rate=learning_rate, epochs=epochs, minibatch_size=32)
print(nn.score(test_x.T, test_y.T))
print(nn.score(x_samples.T, y_samples.T))
print('*' * 50 + 'Not ours' + '*' * 50)

regr = MLPRegressor().fit(x_samples, y_samples.ravel())
print(regr.score(test_x, test_y.ravel()))

plt.plot(errors[20:])
plt.show()
