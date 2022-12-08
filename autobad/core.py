from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict, List, Sequence
from collections import defaultdict
import numpy as np
"""
Simple implementation of eager, reverse-mode and vectorized autodiff. Only dependency is numpy.

No clever handling of batched inputs.

"""


@dataclass
class Tensor:
    value: np.array
    need_grad: Optional[bool] = True
    grad: Optional[np.array] = None


OperatorType = Callable[[np.array, ...], Tuple[np.array, Callable[np.array],
                                               ...]]


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
        graph[id(result)].extend(children)
        return result

    return wrapper_operator


Vjp = Callable[[np.array], np.array]
graph: Dict[Tensor, List[Tuple[Vjp, Tensor]]] = defaultdict(list)

_linear = lambda W, b, x: (np.matmul(W, x) + b, lambda g: g[:, None] *
                           (x * np.ones_like(W)), lambda g: g * np.ones_like(
                               b), lambda g: np.matmul(g, W))
linear = op_wrapper(_linear)

_cos = lambda x: (np.cos(x), lambda g: -np.sin(x) * g)
cos = op_wrapper(_cos)

_mse = lambda x, y: (np.array([(
    (x - y)**2).mean()]), lambda _: 2 / x.shape[0] *
                     (x - y), lambda _: 2 / x.shape[0] * (y - x))

mse = op_wrapper(_mse)
# relu = lambda x: (x[x < 0] = 0; x, lambda: np.array(x<=0))

oned = lambda v: len(v.shape) == 1


def backward(node: Tensor) -> None:
    assert oned(node.value) and node.value.shape[
        0] == 1, f"Sorry this only supports backwards on scalar values. Given shape {node.value.shape}"
    queue: List[Tuple[Tensor, Vjp, Tensor]] = [
        (node, vjp, child) for (vjp, child) in graph[id(node)]
    ]  # (parent, vjp, child)
    while len(queue) != 0:
        parent, vjp, child = queue.pop(0)
        downstream = parent.grad
        if child.grad is not None:
            print("YAY")
        child.grad = vjp(
            downstream) if child.grad is None else child.grad + vjp(downstream)

        queue += [(child, vjp, grandkids)
                  for (vjp, grandkids) in graph[id(child)]]
    graph = defaultdict(list)


np.random.seed(42)
t = Tensor(np.random.rand(10))
yhat = Tensor(np.random.rand(10), need_grad=False)
# print(t.value.shape)
y = cos(t)
l = mse(y, yhat)
# print(l.value.shape)
# print(y)
backward(l)
print(f"T grad")
print(t.grad)
# print(y.grad)

import jax.numpy as jnp
from jax import grad

print("Jax grad")
grad_fn = grad(lambda x: ((yhat.value - jnp.cos(x))**2).mean())
print(grad_fn(t.value))

print("LINEAR layer")
graph = defaultdict(list)
W = Tensor(np.random.rand(5, 10))
b = Tensor(np.random.rand(5))
t = Tensor(np.random.rand(10))
yhat = Tensor(np.random.rand(5), need_grad=False)
y = linear(W, b, t)
# y = relu(y)
mse(y, yhat)
l = mse(y, yhat)
backward(l)
grad_fn = grad(lambda W1, b, t: ((yhat.value - (W1 @ t + b))**2).mean(),
               argnums=[0, 1, 2])
print("jax grad")
print(grad_fn(W.value, b.value, t.value))
print("ours")
print(W.grad)
print(b.grad)
print(t.grad)
# testing multiple paths
# i.e grad wrt. W when y = W((Wx + b1) + b2)
print("Multi test")
graph = defaultdict(list)

# gph = {
#     id(val): (name, val)
#     for (name,
#          val) in [("W", W), ("b1",
#                              b1), ("b2",
#                                    b2), ("t", t), ("yhat",
#                                                    yhat), ("y1",
#                                                            y1), ("y2",
#                                                                  y2), ("l", l)]
# }
print(f"T5: {t.grad}")
for i, neigh in graph.items():
    neigh_names = []
    for (v1, v2) in neigh:
        neigh_names.append(gph[id(v2)][0])
    print(f"{gph[i][0]} {neigh_names}")
from jax import value_and_grad

# pdb.set_trace()
