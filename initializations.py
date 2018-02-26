import numpy as np
import sklearn

def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.sqrt(2 / layers_dims[l-1]) * np.random.randn(layers_dims[l], layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters
