from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from autobad.ops import linear, relu, softmax


def datasets():
    X, y = make_regression(n_samples=10_000)
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    linear()
    softmax()
    relu()
