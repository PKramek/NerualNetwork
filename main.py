import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

from neural_network.layers import Linear, ReLu
from neural_network.loss_functions import MSE
from neural_network.neural_network import NeuralNetwork


def aim_function(x):
    return (x ** 3) / 100


def create_samples(func, num):
    x_s = np.random.uniform(low=-10, high=10, size=(num, 1))
    y_s = np.array([np.array([func(i[0])]) for i in x_s], dtype=np.float32)
    return x_s, y_s


def test_network():
    scores = []
    for i in range(20):
        np.random.seed = i + 1
        x_train, y_train = create_samples(aim_function, 1000)
        x_test, y_test = create_samples(aim_function, 100)
        nn.train(
            x_train.T,
            y_train.T,
            learning_rate=learning_rate,
            epochs=epochs,
            minibatch_size=32,
            calc_test_err=True,
            x_test=x_test.T,
            y_test=y_test.T,
        )
        scores.append(nn.score(x_test.T, y_test.T))
    return np.mean(scores), np.std(scores)


np.random.seed(38)
x_train, y_train = create_samples(aim_function, 1000)

x_test, y_test = create_samples(aim_function, 100)

learning_rate = 0.001
epochs = 200

first = Linear(1, 100)
relu = ReLu(100, 100)
second = Linear(100, 1)

nn = NeuralNetwork(MSE())

second_first = Linear(1, 32)
sec_relu = ReLu(32, 32)
seconds_second = Linear(32, 16)
sec_sec_relu = ReLu(16, 16)
third = Linear(16, 1)

nn.add(first)
nn.add(relu)
nn.add(second)

np.random.seed(42)
print("*" * 50 + "Ours" + "*" * 50)
errors, test_errors = nn.train(
    x_train.T,
    y_train.T,
    learning_rate=learning_rate,
    epochs=epochs,
    minibatch_size=32,
    calc_test_err=True,
    x_test=x_test.T,
    y_test=y_test.T,
)

print(nn.score(x_test.T, y_test.T))

avg_value, std_dev = test_network()
print(f"Average loss function value: {avg_value}")
print(f"Standard devation: {std_dev}")
print("*" * 50 + "Not ours" + "*" * 50)

regr = MLPRegressor().fit(x_train, y_train.ravel())
print(regr.score(x_test, y_test.ravel()))



plt.xlabel("Epoch")
plt.ylabel("Loss function value")
plt.plot(errors, label="Train data")
plt.plot(test_errors, label="Test data")
plt.legend()
plt.grid()
plt.show()
