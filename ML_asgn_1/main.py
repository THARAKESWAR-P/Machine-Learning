import math
from random import random
from tkinter.messagebox import NO
import tree
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_feature_names(path):
    data = pd.read_csv(path)
    return data.columns

def find_best_tree(data):
    best_traindata = None
    best_test_data = None
    best_val_data = None
    Xb_train = None
    Xb_test = None
    Yb_train = None
    Yb_test = None
    Xb_val = None
    Yb_val = None
    best_accuracy = float('inf')
    for i in range(1, 11):
        data = data.sample(frac=1, random_state=i+34)
        
        temp_data = data.sample(frac=.7, random_state=i+34)
        testing_data = data.drop(temp_data.index)

        val_data = temp_data.sample(frac=.2, random_state=i+34)
        training_data = temp_data.drop(val_data.index)

        X_train = training_data.iloc[:, :-1].values
        Y_train = training_data.iloc[:, -1].values.reshape(-1,1)

        X_test = testing_data.iloc[:, :-1].values
        Y_test = testing_data.iloc[:, -1].values.reshape(-1,1)

        X_val = val_data.iloc[:, :-1].values
        Y_val = val_data.iloc[:, -1].values.reshape(-1,1)
        
        temp_regressor = tree.RTRegressor(min_samples_split=15, max_depth=15)
        temp_regressor.fit(X_train,Y_train)
        
        train_acc = temp_regressor.find_accuracy(X_train, Y_train)
        test_acc = temp_regressor.find_accuracy(X_test, Y_test)
        print(f"Training on split {i} complete")
        print(f"Training accuracy: {train_acc}")
        print(f"Testing accuracy: {test_acc}")
        if test_acc < best_accuracy:
            best_accuracy = test_acc
            best_traindata = training_data
            best_test_data = testing_data
            best_val_data = val_data
            Xb_train = X_train
            Xb_test = X_test
            Yb_train = Y_train
            Yb_test = Y_test
            Xb_val = X_val
            Yb_val = Y_val
            best_tree_regressor = temp_regressor

    print("\n"+"+"*50 + "  Best tree accuracies over 10 random splits  " + "+"*50)
    print("Training accuracy:", best_tree_regressor.find_accuracy(Xb_train, Yb_train))
    print("Testing accuracy:", best_tree_regressor.find_accuracy(Xb_test, Yb_test))
    print("Depth: ", best_tree_regressor.find_depth())

    best_tree_regressor.convert_to_gv(feature_names=feature_names, file_name="bestRT.gv")

    best_tree_regressor.pruning(best_traindata, best_test_data)

    best_tree_regressor.convert_to_gv(feature_names=feature_names, file_name="prunedbestRT.gv")


    return best_tree_regressor, Xb_train, Xb_test, Xb_val,  Yb_train, Yb_test, Yb_val


def depth_annalysis(X_train, Y_train,X_test,Y_test ,X_val, Y_val):
    test_dth = []
    train_dth=[]
    val_dth=[]
    node_list = []

    for i in range(1, 20):
        print(f"Checking Height = {i}")
        temp_regressor = tree.RTRegressor(min_samples_split=10, max_depth=i)
        temp_regressor.fit(X_train,Y_train)
        # temp_regressor.convert_to_gv(feature_names=feature_names, file_name=f'{i}.gv')
        
        test_dth.append(temp_regressor.find_accuracy(X_test, Y_test))
        train_dth.append(temp_regressor.find_accuracy(X_train, Y_train))
        val_dth.append(temp_regressor.find_accuracy(X_val, Y_val))
        node_list.append(temp_regressor.count_nodes())

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(range(1, 20), test_dth, label="TESTING")
    axis[0].plot(range(1, 20), train_dth, label="TRAINING")
    axis[0].plot(range(1, 20), val_dth, label="VALIDATION")
    axis[0].set_xlabel("DEPTH")
    axis[0].set_ylabel("ACCURACY")
    axis[0].set_title("ACCURACY vs DEPTH")
    axis[0].legend()

    axis[1].plot(node_list, train_dth, label="TRAINING")
    axis[1].plot(node_list, val_dth, label="VALIDATION")
    axis[1].plot(node_list, test_dth, label="TESTING")
    axis[1].set_xlabel("NODE COUNT")
    axis[1].set_ylabel("ACCURACY")
    axis[1].set_title("ACCURACY vs NODE COUNT")
    axis[1].legend()
    plt.show()

    Optimal_depth = 1+np.argmin(np.array(val_dth))
    best_tree_regressor = tree.RTRegressor(min_samples_split=10, max_depth=Optimal_depth)
    best_tree_regressor.fit(X_train,Y_train)
    print(f"Optimal depth: {Optimal_depth}")
    print(f"NODE COUNT: {best_tree_regressor.count_nodes()}")
    return best_tree_regressor


if __name__ == '__main__':
    
    print("+"*50 + "  START  " + "+"*50)
    feature_names = get_feature_names("Train_B_Tree.csv")

    data = pd.read_csv("Train_B_Tree.csv")
    data.head(20)

    training_data = data.sample(frac=.7 )
    testing_data = data.drop(training_data.index)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    len_train = math.floor(0.7 * (X.shape[0]-1))
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=41)
    #X_train = X[:len_train]
    #Y_train = Y[:len_train]
    #X_test = X[len_train:]
    #Y_test = Y[len_train:]

    X_train = training_data.iloc[:, :-1].values
    Y_train = training_data.iloc[:, -1].values.reshape(-1,1)

    X_test = testing_data.iloc[:, :-1].values
    Y_test = testing_data.iloc[:, -1].values.reshape(-1,1)


    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test:", Y_test.shape)

    regressor = tree.RTRegressor(min_samples_split=15, max_depth=15)
    regressor.fit(X_train,Y_train)

    regressor.print_tree(feature_names=feature_names)
    regressor.convert_to_gv(feature_names=feature_names, file_name="RT.gv")
    print("Testing accuracy: ", regressor.find_accuracy(X_test, Y_test))
    print("Training accuracy; ", regressor.find_accuracy(X_train, Y_train))

    # regressor.pruning(testing_data, training_data)

    print("\n"+"+"*50 + "  Tree accuracies over 10 random splits  " + "+"*50)
    best_tree_regressor, Xb_train, Xb_test, Xb_val,  Yb_train, Yb_test, Yb_val = find_best_tree(data)
    # best_tree_regressor = tree.RTRegressor(min_samples_split=10, max_depth=10)
    # best_tree_regressor.fit(Xb_train,Yb_train)
    print("\n"+"+"*50 + "  Best tree accuracies post pruning  " + "+"*50)
    print("Training accuracy:", best_tree_regressor.find_accuracy(Xb_train, Yb_train))
    print("Testing accuracy:", best_tree_regressor.find_accuracy(Xb_test, Yb_test))
    print("Depth: ", best_tree_regressor.find_depth())

    depth_annalysis(Xb_train, Yb_train,Xb_test,Yb_test ,Xb_val, Yb_val)

    
