import numpy as np
from itertools import chain
import csv
import math


# enumerate data so that they have "keys"

def split_data():
    with open('iris.data', mode='r') as file:
        reader = csv.reader(file)
        Y = []
        X = []
        for row in reader:
            if row:
                if row[-1] == "Iris-setosa":
                    Y.append(0)
                if row[-1] == "Iris-versicolor":
                    Y.append(1)
                if row[-1] == "Iris-virginica":
                    Y.append(2)
                row = [float(x) for x in row[:-1]]
                X.append(row[:-1])

        # randomly shuffle data
        X,Y = np.array(X), np.array(Y)
        idx = np.random.permutation(len(Y))
        X,Y = X[idx], Y[idx]

        # split data and assign keys
        global_train = {}
        train_y = {}
        test_x = {}
        test_y = {}

        train_split = int(0.9*(len(X)))
        x_train = X[0:train_split]
        y_train = Y[0:train_split]
        for idx, x in enumerate(x_train):
            global_train[idx] = x
        for idx, y in enumerate(y_train):
            train_y[idx] = y

        # after making this do not touch!!!
        x_test = X[train_split:]
        y_test = X[train_split:]
        for idx, x in enumerate(x_test):
            test_x[idx] = x
        for idx, y in enumerate(y_test):
            test_y[idx] = y
        
        return global_train, train_y, test_x, test_y


def make_kfold(train_keys, k):
    # split train into 5 equally sized sets
    split_len = len(train_keys) / k
    folds_dic = {}
    for i in range(k):
        folds_dic[i] = train_keys[int(i*split_len) : int((i*split_len)+split_len)]

    folds = [folds_dic[idx] for idx in list(folds_dic.keys())]
    return folds

# implement KNN
def evaluate_model(val_keys, train_keys, train_x, train_y, k):
    ordered_list = []
    prediction = []
    # use numpy to flatten train_keys
    train_keys = list(chain.from_iterable(train_keys))

    validation = [(idx,train_x[idx]) for idx in val_keys]
    train = [(idx, train_x[idx], train_y[idx]) for idx in train_keys]
    # predict for each label in validation (based on train)
    for xi, x in enumerate(validation):
        for yi, y in enumerate(train):
            # compute distances and create a list
            distance = np.linalg.norm(x[1]-y[1])
            ordered_list.append((y[2], distance))

        # sort list by distance
        ordered_list = sorted(ordered_list, key = lambda x: x[1])
        k_neighbors = [y for (y,_) in ordered_list[:k]]

        # make prediction (take max of first k y vals)
        prediction.append((x[0], max(k_neighbors))) # (val_key, pred)

    # evaluate fold performance
    correct = 0
    for pred in prediction:
        if train_y[pred[0]] == pred[1]:
            # accuracte prediction!
            correct += 1

    accuracy = correct / len(prediction) 

    print()
    print("evaluating fold!")
    print(f"fold accuracy: {accuracy:3f}%")
    print()


def main():
    # split data into train and test
    global_train, train_y, test_x, test_y = split_data()
    
    # k-fold
    folds = make_kfold(list(global_train.keys()), 2) 

    # train model on each fold
    for idx,fold in enumerate(folds):
        evaluate_model(fold, (folds[:idx] + folds[idx+1:]), global_train, train_y, 10)
        


if __name__ == "__main__":
    main()
