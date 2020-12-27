from sklearn.neural_network import MLPRegressor
import random


def aim_function(x):
    return (x ** 2) / 10


def create_samples(func):
    x_s = random.sample(range(-100, 100), 100)
    y_s = [func(i) for i in x_s]
    return x_s, y_s


x_samples, y_samples = create_samples(aim_function)

# regr = MLPRegressor(random_state=1)