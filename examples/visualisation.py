import numpy as np

from autobad import linear, mse, relu
from autobad.core import Tensor
from autobad.utils import graph_viz


def render_mlp():

    params = [
        Tensor(np.random.rand(50, 100)),
        Tensor(np.random.rand(50)),
        Tensor(np.random.randn(1, 50)),
        Tensor(np.random.rand(1)),
    ]

    def forward(x: np.array, y: np.array) -> Tensor:
        x = Tensor(x, need_grad=False)
        y = Tensor(y.reshape(1), need_grad=False)
        y0 = linear(params[0], params[1], x)
        y1 = relu(y0)
        y2 = linear(params[2], params[3], y1)
        loss = mse(y, y2)
        print(
            {
                id(params[0]): "W0",
                id(params[1]): "b0",
                id(params[2]): "W1",
                id(params[3]): "b1",
                id(loss): "l",
                id(y0): "y0",
                id(y1): "y1",
                id(y2): "y2",
                id(x): "x",  # This shouldn't appear in the rendering!
                id(y): "y",
                id(x.value): "x",  # This shouldn't appear in the rendering!
                id(y.value): "y",
            }
        )
        graph_viz(
            node_names={
                id(params[0]): "W0",
                id(params[1]): "b0",
                id(params[2]): "W1",
                id(params[3]): "b1",
                id(loss): "l",
                id(y0): "y0",
                id(y1): "y1",
                id(y2): "y2",
            }
        )
        return loss

    forward(np.random.rand(100), np.random.rand(1))


if __name__ == "__main__":
    render_mlp()
