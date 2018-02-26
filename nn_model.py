import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = { "W1": W1, "b1": b1, "W2": W2, "b2": b2 }
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = { "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2 }
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    cost = np.sum(np.multiply(np.log(A2), Y))
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads

def update_parameters(parameters, grads, learning_rate = 0.09):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    W1 = W1 - (learning_rate * dW1)
    W2 = W2 - (learning_rate * dW2)
    b1 = b1 - (learning_rate * db1)
    b2 = b2 - (learning_rate * db2)

    parameters = { "W1": W1, "b1": b1, "W2": W2, "b2": b2 }
    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

def nn_model(X_train, Y_train, X_test, Y_test, n_h, num_iterations = 1000, learning_rate = 0.09, print_cost = False):
    n_x = layer_sizes(X_train, Y_train)[0]
    n_y = layer_sizes(X_train, Y_train)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A2, Y_train, parameters)
        grads = backward_propagation(parameters, cache, X_train, Y_train)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration: ", i, cost)

    Y_prediction_train = predict(parameters, X_train)
    Y_prediction_test = predict(parameters, X_test)
    d = {
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return d
