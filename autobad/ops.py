import numpy as np
from typing import Sequence
from autobad.typing import OperatorType
from autobad.core import Tensor
from autobad.graph import Graph

eps = 1e-9


def op_wrapper(op: OperatorType):
    """
        Purpose of this decorator is to modify the graph structure 
        when an operator is called.

        Suppose c= op(a,b), we add edges a->c and b->c to the graph.
        The edge contains the vector Jacobian product.
    """

    def wrapper_operator(*inputs: Sequence[Tensor]):
        values = [i.value for i in inputs]
        output = op(*values)
        result, vjps = output[0], output[1:]
        children = [(v, i) for (v, i) in zip(vjps, inputs) if i.need_grad]
        result = Tensor(result, need_grad=len(children) > 0)
        Graph.add(result, children)
        return result

    return wrapper_operator


_linear = lambda W, b, x: (np.matmul(W, x) + b, lambda g: g[:, None] *
                           (x * np.ones_like(W)), lambda g: g * np.ones_like(
                               b), lambda g: np.matmul(g, W))
linear = op_wrapper(_linear)

_cos = lambda x: (np.cos(x), lambda g: -np.sin(x) * g)
cos = op_wrapper(_cos)

_mse = lambda x, y: (((x - y)**2).mean().reshape(1), lambda _: 2 / x.shape[0] *
                     (x - y), lambda _: 2 / x.shape[0] * (y - x))
mse = op_wrapper(_mse)

_sin = lambda x: (np.sin(x), lambda g: np.cos(x) * g)
sin = op_wrapper(_sin)

_relu = lambda x: (np.maximum(0, x), lambda g: g * (x >= 0))
relu = op_wrapper(_relu)


def _softmax_forward(x: np.array) -> np.array:
    s = np.exp(x - np.max(x))
    result = s / s.sum()
    return result, lambda g: np.matmul(
        g,
        np.diag(result) - np.outer(result, result))


_softmax = lambda x: _softmax_forward(x)
softmax = op_wrapper(_softmax)

_cross_entropy = lambda yhat, y: (-np.sum(yhat * np.log(y + eps)).reshape(
    1), lambda _: -np.log(y + eps), lambda _: -yhat / (y + eps))
cross_entropy = op_wrapper(_cross_entropy)