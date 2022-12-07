from sklearn import datasets
import numpy as np
from autobad.core import linear, mse, relu, Tensor

n_features = 100
X, y = datasets.make_regression(n_features=n_features)
W = Tensor(value=np.random.uniform()))
