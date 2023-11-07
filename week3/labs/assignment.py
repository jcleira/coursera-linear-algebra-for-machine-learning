import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Set a seed so that the results are consistent.
np.random.seed(3)

X = np.array([[ 0.3190391, -1.07296862, 0.86540763, -0.17242821, 1.14472371, 0.50249434, -2.3015387, -0.68372786, -0.38405435, -0.87785842, -2.06014071, -1.10061918, -1.09989127, 1.13376944, 1.74481176, -0.12289023, -0.93576943, 1.62434536, 1.46210794, 0.90159072, -0.7612069, 0.53035547, -0.52817175, -0.26788808, 0.58281521, 0.04221375, 0.90085595, -0.24937038, -0.61175641, -0.3224172 ]])


Y = np.array([[ -3.01854669, -65.65047675, 26.96755728, 8.70562603, 57.94332628, -0.69293498, -78.66594473, -12.73881492, -13.26721663, -24.80488085, -74.24484385, -39.99533724, -22.70174437, 73.46766345, 55.7257405, 23.80417646, -13.45481508, 25.57952246, 75.91238321, 50.91155323, -43.7191551, -1.7025559, -16.44931235, -33.54041234, 20.4505961, 18.35949302, 37.69029586, -1.04801683, -4.47915933, -20.89431647]])


shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))


def initialize_parameters(n_x, n_y):
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))

    parameters = {"W": W,
                  "b": b}
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))
