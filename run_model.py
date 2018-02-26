import numpy as np
import json
import sklearn
import sklearn.linear_model
# import matplotlib.pyplot as plt
from parse_csv import parse_csv
from logistic_model import logistic_model
from nn_model_2 import two_layer_model, L_layer_model
from nn_model_3 import model_with_initialization
from nn_model_4 import model_with_regularization, model_with_dropout
from nn_model import nn_model

xle_file = "datasets/XLE.csv" # training data for Energy sector - best with 250 day lookback (58% accuracy)
vnq_file = "datasets/VNQ.csv" # training data for REIT sector - best with 250 day lookback (68% accuracy)
xlb_file = "datasets/XLB.csv" # training data for XLB - Materials sector - best with 250 day lookback (55% accuracy)
xlu_file = "datasets/XLU.csv" # training data for XLU - Utilities sector - best with 250 day lookback (FILE CORRUPTED)
vox_file = "datasets/VOX.csv" # training data for VOX - Telecom sector - best with 250 day lookback (44% accuracy)
xlk_file = "datasets/XLK.csv" # training data for XLK - Technology sector
xlf_file = "datasets/XLF.csv" # training data for XLF - Financials sector
xlv_file = "datasets/XLV.csv" # training data for Health Care sector
xly_file = "datasets/XLY.csv" # training data for Consumer Discretionary sector
xlp_file = "datasets/XLP.csv" # training data for Consumer Staples sector
xli_file = "datasets/XLI.csv" # training data for Industrials sector

X_train, Y_train, X_test, Y_test = parse_csv(xlv_file, 70, 1900)
print("X, Y: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

## LOGISTIC MODEL (COURSERA)
# d = logistic_model(X_train, Y_train, X_test, Y_test, num_iterations = 100, learning_rate = 0.015, print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (logistic model): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (logistic model): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')

### LOGISTIC MODEL (SKLEARN)
clf = sklearn.linear_model.LogisticRegressionCV()
Y_1d = Y_train.reshape(Y_train.shape[1],)
clf.fit(X_train.T, Y_1d.T)

LR_predictions_train = clf.predict(X_train.T)
LR_predictions_test = clf.predict(X_test.T)
print('Test Accuracy (sklearn logistic model): %d' % float((np.dot(Y_test, LR_predictions_test.T) + np.dot(1 - Y_test, 1 - LR_predictions_test.T)) / float(Y_test.size)*100) + '%')
print('Train Accuracy (sklearn logistic model): %d' % float((np.dot(Y_train, LR_predictions_train.T) + np.dot(1-Y_train,1-LR_predictions_train.T))/float(Y_train.size)*100) + '%')

### SIMPLE NEURAL NET (COURSERA)
# n_x = 4
# d = nn_model(X_train, Y_train, X_test, Y_test, n_x, num_iterations = 100, learning_rate = 0.09, print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (2 layer net): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (2 layer net): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')

### TWO LAYER NEURAL NET (COURSERA)
# n_x = X_train.shape[0]
# n_h = 4
# n_y = Y_train.shape[0]
# layers_dims = (n_x, n_h, n_y)
# d = two_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations = 100, learning_rate = 0.09, print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (2 layer net): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (2 layer net): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')

#### 5 LAYER NEURAL NET (COURSERA)
# layers_dims = [X_train.shape[0], 20, 7, 10, 1] #  5-layer model
# d = L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations = 100, learning_rate = 0.09, initialization = "zeros", print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (zero initialization): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (zero initialization): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')

#### 5 LAYER NET WTIH DIFFERENT INITIALIZATIONS
layers_dims = [X_train.shape[0], 20, 7, 10, 1] #  5-layer model
# d = model_with_initialization(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations = 100, learning_rate = 0.09, initialization = "random", print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (random initialization): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (random initialization): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')

d = model_with_initialization(X_train, Y_train, X_test, Y_test, layers_dims, num_iterations = 3000, learning_rate = 0.05, initialization = "he", print_cost = True)
Y_prediction_test = d["Y_prediction_test"]
Y_prediction_train = d["Y_prediction_train"]
print('Test Accuracy (He initialization): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
print('Train Accuracy (He initialization): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')
print("Params: ", d)

f = {
 "W1": d["parameters"]["W1"].tolist(),
 "b1": d["parameters"]["b1"].tolist(),
 "W2": d["parameters"]["W2"].tolist(),
 "b2": d["parameters"]["b2"].tolist(),
 "W3": d["parameters"]["W3"].tolist(),
 "b3": d["parameters"]["b3"].tolist(),
 "W4": d["parameters"]["W4"].tolist(),
 "b4": d["parameters"]["b4"].tolist(),
}

print("F: ", f)

with open('data.json', 'w') as outfile:
    json.dump(f, outfile)
# layers_dims = [X_train.shape[0], 20, 3, 1]
# d = model_with_regularization(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.09, num_iterations = 2000, lambd = 0.7, print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (Lambda regularization): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (Lambda regularization): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')
#
# d = model_with_dropout(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.3, num_iterations = 2000, keep_prob = 0.5, print_cost = False)
# Y_prediction_test = d["Y_prediction_test"]
# Y_prediction_train = d["Y_prediction_train"]
# print('Test Accuracy (Dropout regularization): %d' % float((np.dot(Y_test, Y_prediction_test.T) + np.dot(1 - Y_test, 1 - Y_prediction_test.T)) / float(Y_test.size)*100) + '%')
# print('Train Accuracy (Dropout regularization): %d' % float((np.dot(Y_train, Y_prediction_train.T) + np.dot(1-Y_train,1-Y_prediction_train.T))/float(Y_train.size)*100) + '%')
