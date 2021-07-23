from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']


def show_some_digit(X):
    sample = np.random.randint(X.shape[0])
    digit_image = X[sample].reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()

show_some_digit(X)