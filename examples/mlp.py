from typing import List

import more_itertools
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from autobad import Tensor, backward, linear, mean, mse, relu, sgd
from autobad.utils import graph_viz


def train():
    n_samples = 1_000
    X, y = make_regression(n_samples=n_samples, n_features=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    params = [
        Tensor(np.random.rand(50, 100)),
        Tensor(np.random.rand(50)),
        Tensor(np.random.randn(1, 50)),
        Tensor(np.random.rand(1)),
    ]
    n_epochs = 100
    batch_size = 16
    train_losses = []

    def forward(x: np.array, y: np.array) -> float:
        x = Tensor(x, need_grad=False)
        y = Tensor(y.reshape(1), need_grad=False)
        y0 = linear(params[0], params[1], x)
        y1 = relu(y0)
        y2 = linear(params[2], params[3], y1)
        return mse(y, y2)

    for n_epoch in range(n_epochs):
        indxs = np.random.permutation(np.arange(len(X_train)))
        batches = more_itertools.chunked(indxs, n=batch_size)
        batch_losses = []
        for indxs in batches:
            samples_losses = []
            for (x, y) in zip(X_train[indxs], y_train[indxs]):
                samples_losses.append(forward(x, y))

            batch_loss = mean(*samples_losses)
            if n_epoch == 0:
                graph_viz(
                    {
                        id(params[0]): "W0",
                        id(params[1]): "b0",
                        id(params[2]): "W1",
                        id(params[3]): "b1",
                        id(batch_loss): "batch_loss",
                    }
                )

            backward(batch_loss)
            sgd(params, lr=1e-3)
            batch_losses.append(batch_loss.value[0])
        train_losses.append(sum(batch_losses) / len(batch_losses))
    print(train_losses[-15:])

    test_loss = np.array(
        [forward(x, y).value[0] for (x, y) in zip(X_test, y_test)]
    ).mean()
    print(f"{test_loss=}")
    df = pd.DataFrame({"Epoch": np.arange(len(train_losses)), "MSE": train_losses})
    fig = px.line(df, x="Epoch", y="MSE")
    fig.show()


if __name__ == "__main__":
    train()
