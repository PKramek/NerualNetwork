from sklearn.neural_network import MLPRegressor
import numpy as np
import random


def aim_function(x):
    return (x ** 2) / 10


def create_samples(func, num):
    x_s = np.random.uniform(low=-10, high=10, size=(num, 1))
    y_s = [func(i[0]) for i in x_s]
    return x_s, y_s


x_samples, y_samples = create_samples(aim_function, 10)

test_x, test_y = create_samples(aim_function, 10)

regr = MLPRegressor(random_state=1).fit(x_samples, y_samples)

print(regr.score(test_x, test_y))
