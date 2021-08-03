from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
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

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X[shuffle_index], y[shuffle_index]

def prepare_data():
    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'].astype(np.int8), mnist['target'].astype(np.int8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X[shuffle_index], y[shuffle_index]
    return  X_train, X_test, y_train, y_test

def use_SGDClassifier(X_train, X_test, y_train, y_test):
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_train, y_train)
    y_train_predict = cross_val_score(sgd_clf, X_train, y_train, cv=3)
    print('SGDClassifier results:')
    print('Cross validation score:')
    print(y_train_predict)
    y_test_predict = sgd_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    print('Confusion matrix:')
    print(conf_matrix)
    accuracy = accuracy_score(y_test, y_test_predict)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_test, y_test_predict)
    print('Precision: ', precision)
    recall = recall_score(y_test, y_test_predict)
    print('Recall: ', recall)
    clf_f1_score = f1_score(y_test, y_test_predict)
    print('f1 score: ', clf_f1_score, '\n\n')
    return sgd_clf

def use_RandomForestClassifier(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    y_train_predict = cross_val_score(rf_clf, X_train, y_train, cv=3)
    print('RandomForestClassifier results:')
    print('Cross validation score:')
    print(y_train_predict)
    y_test_predict = rf_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    print('Confusion matrix:')
    print(conf_matrix)
    accuracy = accuracy_score(y_test, y_test_predict)
    print('Accuracy: ', accuracy)
    precision = precision_score(y_test, y_test_predict)
    print('Precision: ', precision)
    recall = recall_score(y_test, y_test_predict)
    print('Recall: ', recall)
    clf_f1_score = f1_score(y_test, y_test_predict)
    print('f1 score: ', clf_f1_score, '\n\n')
    return rf_clf


X_train, X_test, y_train, y_test = prepare_data()
use_SGDClassifier(X_train, X_test, y_train, y_test)


'''
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


sgd_clf = SGDClassifier( random_state=42)
sgd_clf.fit(X_train, y_train_5)
y_pred_5 = sgd_clf.predict(X_test)
y_train_5_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
conf_matrix = confusion_matrix(y_train_5, y_train_5_predict)
print('confusion matrix:\n', conf_matrix)
print('precission: ', precision_score(y_train_5, y_train_5_predict))
print('recall: ', recall_score(y_train_5, y_train_5_predict))
print('f1 score: ', f1_score(y_train_5, y_train_5_predict))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')


fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

forrest_clf = RandomForestClassifier(random_state=42)
y_probas_forrest = cross_val_predict(forrest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forrest = y_probas_forrest[:, 1]

fpr_forrest, tpr_forrest, thresholds_forrest = roc_curve(y_train_5, y_scores_forrest)
plt.plot(fpr, tpr, 'b:', label='GSD')
plot_roc_curve(fpr_forrest, tpr_forrest, 'Random forrest')
plt.legend(loc='lower right')
plt.show()
'''