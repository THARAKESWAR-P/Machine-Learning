from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np


if __name__ == '__main__':

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
    label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    df['target'].replace(label_map,inplace = True)

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # Separating out the features
    
    for feature in features:
        mean, sigma = df[feature].mean(), df[feature].std()
        df[feature] = (df[feature]-mean)/sigma

    dataset = np.array(df)
    x = dataset[:, :-1]
    y = dataset[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    linear_clf = svm.SVC(kernel='linear')
    linear_clf = linear_clf.fit(x_train, y_train)
    y_pred = linear_clf.predict(x_test)
    print("\n"+"*"*50 + "  SVM ACCURACIES  " + "*"*50+"\n")
    accuracy = np.sum((y_pred == y_test)/len(y_test))
    print("Linear function SVM accuracy :- ", accuracy)

    poly_clf = svm.SVC(kernel='poly')
    poly_clf = poly_clf.fit(x_train, y_train)
    y_pred = poly_clf.predict(x_test)
    accuracy = np.sum((y_pred == y_test)/len(y_test))
    print("Quadratic function SVM accuracy :- ", accuracy)

    radial_clf = svm.SVC(kernel='rbf')
    radial_clf = radial_clf.fit(x_train, y_train)
    y_pred = radial_clf.predict(x_test)
    accuracy = np.sum((y_pred == y_test)/len(y_test))
    print("RBF function SVM accuracy :- ", accuracy)

    print("\n"+"*"*50 + "  MLP CLASSIFIER  " + "*"*50+"\n")
    mlp_clf1 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.001, hidden_layer_sizes=(16, ))
    mlp_clf1.fit(x_train, y_train)
    y_pred = mlp_clf1.predict(x_test)
    print("Accuracy of MLP classifier for 1 hidden layer and 16 nodes :- ", np.sum((y_pred == y_test)/len(y_test)))
    mlp_clf2 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.001, hidden_layer_sizes=(256, 16))
    mlp_clf2.fit(x_train, y_train)
    y_pred = mlp_clf2.predict(x_test)
    print("Accuracy of MLP classifier for 2 hidden layers - 256 and 16 respectively :- ", np.sum((y_pred == y_test)/len(y_test)))

    print("\n"+"*"*50 + "  LEARNING RATE vs ACCURACY  " + "*"*50+"\n")
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    mlp_accuracies = []
    for lr in learning_rates:
        mlp_clf = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=lr, hidden_layer_sizes=(256, 16))
        mlp_clf.fit(x_train, y_train)
        y_pred = mlp_clf.predict(x_test)
        mlp_accuracies.append(np.sum((y_pred == y_test)/len(y_test)))

    plt.plot(learning_rates, mlp_accuracies)
    plt.xlabel('Learning rates')
    plt.ylabel('Accuracy')
    plt.title('LEARNING RATE vs ACCURACY')
    plt.show()

    print('\nLearning rates :- ', learning_rates)
    print('\nAccuracy of MLP classifier for above learning rates(respectively):- ', mlp_accuracies, '\n\n')


    print("\n"+"*"*50 + "  BAcKWARD ELIMINATION METHOD PROCESSING  " + "*"*50 + "\n")
    bfs = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1), k_features=(1, 4), forward=False, floating=False, verbose=2, scoring='accuracy', cv=5).fit(x_train, y_train)
    print("\n\n"+"*"*50 + "  BEST FEATURES  " + "*"*50 + "\n")
    # print(bfs.k_feature_names_)
    for id in bfs.k_feature_names_:
        print(features[int(id)-1])
    clf_1 = svm.SVC(kernel='poly', degree=2)
    clf_2 = svm.SVC(kernel='rbf', degree=2)
    clf_3 = MLPClassifier(batch_size=32, solver='lbfgs', learning_rate_init=0.1, hidden_layer_sizes=(256, 16))

    ensemble_clf = EnsembleVoteClassifier(clfs=[clf_1, clf_2, clf_3], weights=[1, 1, 1])
    ensemble_clf.fit(x_train, y_train)
    y_pred = ensemble_clf.predict(x_test)
    print("\n"+"*"*50 + "  MAX-VOTE CLASSIFIER  " + "*"*50+"\n")
    print('Accuracy of Max-vote classifier :- ', np.sum((y_pred == y_test)/len(y_test)), '\n\n')



    

