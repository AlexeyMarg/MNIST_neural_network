from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)