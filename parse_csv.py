import csv
import numpy as np

def parse_csv(csvfile, lookback = 250, threshold = 1800):
    X = [ ] # initialize array of all training examples
    Y = [ ] # initialize array of all training labels
    X_train = [ ] # initialize training examples
    Y_train = [ ] # initialize training labels
    X_test = [ ] # initialize test examples
    Y_test = [ ] # initialize test labels
    rows = [ ]
    with open(csvfile, newline='') as file:
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for idx, row in enumerate(reader):
            if idx is 1:
                line = (', ').join(row)
                data = np.array(line.split(',')[1:]).astype(np.float)
                print("Data: ", data.shape)
            if idx is not 0:
                line = (', ').join(row)
                data = np.array(line.split(',')[1:]).astype(np.float)

                rows.append(data[:-1]) # do not store date
                y = float(data[-1])
                y = 1 if y > 0 else 0
                Y.append(y)

    m = len(rows) - lookback

    num_features = len(rows[0]) + lookback - 1

    for idx, row in enumerate(rows):
        if idx >= lookback:
            example = np.array(rows[idx-lookback:idx])
            # only keep
            # print("Example: ", example.shape)
            w = example.shape[0]
            h = example.shape[1]
            lb = example[:-1,h-1]
            bottom = example[h-1,:]
            x = np.append(lb, bottom)
            x = np.reshape(x, (1, num_features))
            X.append(x)

    Y = np.array(Y[lookback:])

    X = np.array(X)

    X = np.reshape(X, (num_features, m))
    # normalize data -
    # One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array.
    # But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
    # X = X - np.mean(X)
    # X = X / np.std(X)

    Y = np.reshape(Y, (1, m))

    X_train = X[:,0:threshold]
    Y_train = Y[:,0:threshold]

    X_test = X[:, threshold:-1]
    Y_test = Y[:, threshold:-1]

    # split into train and test
    return X_train, Y_train, X_test, Y_test
