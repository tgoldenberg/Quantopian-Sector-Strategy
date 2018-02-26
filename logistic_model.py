from parse_csv import parse_csv
import numpy as np
import sklearn
import sklearn.linear_model

def sigmoid(z): # sigmoid activation function
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim): # initialize weights (W) and bias (b) params
    W = np.zeros((dim, 1))
    b = 0
    return W, b

def propagate(W, b, X, Y): # given (W, b) predict values for X
    m = X.shape[1] # number of examples
    A = sigmoid(np.dot(W.T, X) + b)
    cost = -1/m * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))
    dW = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = { "dW": dW, "db": db }
    return grads, cost

def optimize(W, b, X, Y, num_iterations, learning_rate, print_cost = False): # given derivatives of W & b, move closer to zero cost
    costs = [ ]
    for i in range(num_iterations):
        grads, cost = propagate(W, b, X, Y)
        dW = grads["dW"]
        db = grads["db"]
        W = W - (learning_rate * dW)
        b = b - (learning_rate * db)
        costs.append(cost)
        if print_cost and i % 1000 == 0:
            print('Cost after iteration: ', i, cost)
    params = { "W": W, "b": b }
    grads = { "dW": dW, "db": db }
    return params, grads, costs

def predict(W, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(W.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction

def logistic_model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    W, b = initialize_with_zeros(X_train.shape[0])
    grads, cost = propagate(W, b, X_train, Y_train)
    parameters, grads, costs = optimize(W, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    W = parameters["W"]
    b = parameters["b"]
    Y_prediction_test = predict(W, b, X_test)
    Y_prediction_train = predict(W, b, X_train)

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "W": W,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return d
