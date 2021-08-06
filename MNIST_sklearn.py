from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def show_some_digit(X):
    sample = np.random.randint(X.shape[0])
    digit_image = X[sample].reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()

#show_some_digit(X)

def prepare_data():
    print('Loading of data')
    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'].astype(np.int8), mnist['target'].astype(np.int8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X[shuffle_index], y[shuffle_index]
    return  X_train, X_test, y_train, y_test

def use_SGDClassifier(X_train, X_test, y_train, y_test):
    sgd_clf = SGDClassifier()
    print('\n\nStsrt fit SGD\n\n')
    sgd_clf.fit(X_train, y_train)
    y_train_cv_score = cross_val_score(sgd_clf, X_train, y_train, cv=3)
    print('SGDClassifier results for test data:')
    print('Cross validation score:')
    print(y_train_cv_score)
    print('\nTest data results')
    y_test_predict = sgd_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    print('Confusion matrix:')
    print(conf_matrix)
    accuracy = accuracy_score(y_test, y_test_predict)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_test, y_test_predict, average=None)
    print('Precision: ', precision)
    recall = recall_score(y_test, y_test_predict, average=None)
    print('Recall: ', recall)
    clf_f1_score = f1_score(y_test, y_test_predict, average=None)
    print('f1 score: ', clf_f1_score, '\n\n')
    return sgd_clf

def use_RandomForestClassifier(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier()
    print('\n\nStsrt fit RF\n\n')
    rf_clf.fit(X_train, y_train)
    y_train_cv = cross_val_score(rf_clf, X_train, y_train, cv=3)
    print('RandomForestClassifier results for test data:')
    print('Cross validation score:')
    print(y_train_cv)
    print('\nTest data results')
    y_test_predict = rf_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    print('Confusion matrix:')
    print(conf_matrix)
    accuracy = accuracy_score(y_test, y_test_predict)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_test, y_test_predict, average=None)
    print('Precision: ', precision)
    recall = recall_score(y_test, y_test_predict, average=None)
    print('Recall: ', recall)
    clf_f1_score = f1_score(y_test, y_test_predict, average=None)
    print('f1 score: ', clf_f1_score, '\n\n')
    return rf_clf

def prediction(clf, X):
    y = clf.predict()
    return y



X_train, X_test, y_train, y_test = prepare_data()

sgd_clf = use_SGDClassifier(X_train, X_test, y_train, y_test)

rf_clf = use_RandomForestClassifier(X_train, X_test, y_train, y_test)

