import numpy as np
# import matplotlib.pyplot as plt
from initializations import initialize_parameters_zeros, initialize_parameters_random, initialize_parameters_he
from regularization import compute_cost_with_regularization, backward_propagation_with_regularization, forward_propagation_with_dropout, backward_propagation_with_dropout
# import h5py

def compute_cost(AL, Y):
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def forward_propagation(X, parameters):
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    # Forward propagation
    probas, caches = forward_propagation(X, parameters)
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    # print("Accuracy: "  + str(np.sum((p == y)/m)))
    return p



def model_with_regularization(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0075, num_iterations = 3000, lambd = 0.7, print_cost = False):
    costs = []                         # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_random(layers_dims)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X_train, parameters)
        # Compute cost.
        cost = compute_cost_with_regularization(a3, Y_train, parameters, lambd)

        grads = backward_propagation_with_regularization(X_train, Y_train, cache, lambd)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    Y_prediction_train = predict(X_train, Y_train, parameters)
    Y_prediction_test = predict(X_test, Y_test, parameters)
    d = {
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "parameters": parameters,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return d

def model_with_dropout(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.3, num_iterations = 3000, keep_prob = 0.86, print_cost = False):
    grads = {}
    costs = []                         # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_random(layers_dims)
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        a3, cache = forward_propagation_with_dropout(X_train, parameters, keep_prob)
        # Compute cost.
        cost = compute_cost(a3, Y_train)

        grads = backward_propagation_with_dropout(X_train, Y_train, cache, keep_prob)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    Y_prediction_train = predict(X_train, Y_train, parameters)
    Y_prediction_test = predict(X_test, Y_test, parameters)
    d = {
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "parameters": parameters,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return d
