from typing import Optional, Sequence, Tuple

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
        result, vjps = op(*values)
        children = [(v, i) for (v, i) in zip(vjps, inputs) if i.need_grad]
        tensor = Tensor(result, need_grad=len(children) > 0)
        if tensor.need_grad:
            Graph.add(tensor, children)
        return tensor

    return wrapper_operator


@op_wrapper
def linear(W: np.array, b: np.array, x: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    return (
        np.matmul(W, x) + b,
        [
            lambda g: g[:, None] * (x * np.ones_like(W)),
            lambda g: g * np.ones_like(b),
            lambda g: np.matmul(g, W),
        ],
    )


@op_wrapper
def cos(x: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    return np.cos(x), [lambda g: -np.sin(x) * g]


@op_wrapper
def sin(x: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    return np.sin(x), [lambda g: np.cos(x) * g]


@op_wrapper
def relu(x: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    return np.maximum(0, x), [lambda g: g * (x >= 0)]


@op_wrapper
def softmax(x: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    s = np.exp(x - np.max(x))
    result = s / s.sum()
    return result, [lambda g: np.matmul(g, np.diag(result) - np.outer(result, result))]


def safe_downstream(downstream: Optional[np.array], shape: Tuple[int]) -> np.array:
    """Used to protect against cases where the incoming gradient may feasibly not exist. E.g. for a loss function."""
    return np.ones(shape) if downstream is None else downstream


@op_wrapper
def cross_entropy(y_true: np.array, y_pred: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    """Here we assume there is no downstream gradient."""
    return (
        -np.sum(y_true * np.log(y_pred + eps)).reshape(1),
        [
            lambda g: -safe_downstream(g, y_true.shape) * np.log(y_pred + eps),
            lambda g: -safe_downstream(g, y_true.shape) * (y_true / (y_pred + eps)),
        ],
    )


@op_wrapper
def mse(x: np.array, y: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    """Function is symmetric, hence the ambigious parameter names."""
    return (
        ((x - y) ** 2).mean().reshape(1),
        [
            lambda g: safe_downstream(g, x.shape) * 2 / x.shape[0] * (x - y),
            lambda g: safe_downstream(g, x.shape) * 2 / x.shape[0] * (y - x),
        ],
    )


@op_wrapper
def mean(*inputs: np.array) -> Tuple[np.array, Sequence[Vjp]]:
    """Assumes all the shapes are the same."""
    return np.mean(np.vstack(inputs), axis=0), [
        lambda g: 1 / len(inputs) * safe_downstream(g, inputs[0].shape) for _ in inputs
    ]
