from typing import List
from sklearn.datasets import make_regression
import more_itertools
from sklearn.model_selection import train_test_split

from autobad import Tensor, linear, relu, softmax, sgd, mse, backward, mean
import numpy as np
import matplotlib.pyplot as plt


def train():
    n_samples = 1_000
    X, y = make_regression(n_samples=n_samples, n_features=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    params = [
        Tensor(np.random.rand(50, 100)),
        Tensor(np.random.rand(50)),
        Tensor(np.random.randn(1, 50)),
        Tensor(np.random.rand(1))
    ]
    n_epochs = 400
    batch_size = 512
    train_losses = []

    def running_mean(params: List[Tensor], count: int) -> None:
        # p.grad contains
        for p in params:
            p.grad = p.grad / (count - 1)

    def forward(x: np.array, y: np.array) -> float:
        x = Tensor(x, need_grad=False)
        y = Tensor(y.reshape(1), need_grad=False)
        y0 = linear(params[0], params[1], x)
        y1 = relu(y0)
        y2 = linear(params[2], params[3], y1)
        return mse(y, y2)

    for _ in range(n_epochs):
        indxs = np.random.permutation(np.arange(len(X_train)))
        batch_indxs = more_itertools.chunked(indxs, n=batch_size)
        losses = []
        for (x, y) in zip(X_train[batch_indxs], y_train[batch_indxs]):
            losses.append(forward(x, y))

        mean_loss = mean(*losses)
        backward(mean_loss)
        train_losses.append(mean_loss.value[0])

        sgd(params, lr=1e-2, batch_size=batch_size)
    print(train_losses[-15:])

    test_loss = np.array(
        [forward(x, y).value[0] for (x, y) in zip(X_test, y_test)]).mean()
    print(test_loss)
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.show()


if __name__ == "__main__":
    train()