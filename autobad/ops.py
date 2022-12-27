from typing import Sequence, Tuple

import numpy as np

from autobad.core import Tensor
from autobad.graph import Graph
from autobad.typing import OperatorType, Vjp

eps = 1e-9


def op_wrapper(op: OperatorType):
    """
    Purpose of this decorator is to modify the graph structure
    when an operator is called. It also allows us to defined ops
    based on np arrays, rather than calling Tensor.value.

    Suppose c= op(a,b), we add edges c->a and c->b to the graph.
    The edge is the vector Jacobian product, i.e. an edge, c->b,
    is the function g*dc/db where g is the incoming gradient.
    This might be the gradient of a loss wrt. c for example.
    """

    def wrapper_operator(*inputs: Sequence[Tensor]):
        values = [i.value for i in inputs]
        output = op(*values)
        result, vjps = output[0], output[1:]
        children = [(v, i) for (v, i) in zip(vjps, inputs) if i.need_grad]
        result = Tensor(result, need_grad=len(children) > 0)
        if result.need_grad:
            Graph.add(result, children)
        return result

    return wrapper_operator


@op_wrapper
def linear(W: np.array, b: np.array, x: np.array) -> Tuple[np.array, Vjp, Vjp, Vjp]:
    return (
        np.matmul(W, x) + b,
        lambda g: g[:, None] * (x * np.ones_like(W)),
        lambda g: g * np.ones_like(b),
        lambda g: np.matmul(g, W),
    )


@op_wrapper
def cos(x: np.array) -> Tuple[np.array, Vjp]:
    return np.cos(x), lambda g: -np.sin(x) * g


@op_wrapper
def sin(x: np.array) -> Tuple[np.array, Vjp]:
    return np.sin(x), lambda g: np.cos(x) * g


@op_wrapper
def relu(x: np.array) -> Tuple[np.array, Vjp]:
    return np.maximum(0, x), lambda g: g * (x >= 0)


@op_wrapper
def softmax(x: np.array) -> Tuple[np.array, Vjp]:
    s = np.exp(x - np.max(x))
    result = s / s.sum()
    return result, lambda g: np.matmul(g, np.diag(result) - np.outer(result, result))


@op_wrapper
def cross_entropy(y_true: np.array, y_pred: np.array) -> Tuple[np.array, Vjp, Vjp]:
    """Here we assume there is no downstream gradient."""
    return (
        -np.sum(y_true * np.log(y_pred + eps)).reshape(1),
        lambda _: -np.log(y_pred + eps),
        lambda _: -y_true / (y_pred + eps),
    )


@op_wrapper
def mse(x: np.array, y: np.array) -> Tuple[np.array, Vjp, Vjp]:
    """Symmetric."""
    return (
        ((x - y) ** 2).mean().reshape(1),
        lambda _: 2 / x.shape[0] * (x - y),
        lambda _: 2 / x.shape[0] * (y - x),
    )
