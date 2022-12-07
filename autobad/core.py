from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict, List
from collections import defaultdict
import numpy as np
"""
Simple implementation of eager, reverse-mode and vectorized autodiff. Only dependency is numpy.

No clever handling of batched inputs.

"""


@dataclass
class Tensor:
    value: np.array
    need_grad: bool = True
    grad: Optional[np.array] = None


Vjp = Callable[[np.array], np.array]
graph: Dict[Tensor, List[Tuple[Vjp, Tensor]]] = defaultdict(list)

linear = lambda W, b, x: (np.matmul(W, x) + b, lambda: x * np.ones_like(W),
                          lambda: np.ones_like(b), lambda: np)
cos = lambda x: (np.cos(x), lambda grad: -grad * np.sin(x))

# vectorized mse is  m = 1/N*(x-y)^T(x-y) = 1/N * (x^Tx + y^Ty - 2x^Ty)
# note that x^Ty = y^Tx
# dm/dx = 1/N(2x - 2y) and dm/dy = 1/N(2y-2x)
# this mse implementation only works for vector inputs
mse = lambda x, y: (np.array([((x - y)**2).mean()]), lambda: 2 / x.shape[0] *
                    (x - y), lambda: 2 / x.shape[0] * (y - x))

# relu = lambda x: (x[x < 0] = 0; x, lambda: np.array(x<=0))

oned = lambda v: len(v.shape) == 1
twod = lambda v: len(v.shape) == 2


def vjp_wrapper(vjp):
    def new_vjp(grad: np.array) -> np.array:
        v = vjp()
        if grad is None:
            return v
        grad = grad.reshape(-1, 1) if oned(grad) and twod(v) else grad
        return grad * v

    return new_vjp


def op_wrapper(op):
    def wrapper_operator(*inputs):
        values = [i.value for i in inputs]
        result, vjps = (r := op(*values))[0], (vjp_wrapper(v) for v in r[1:])
        children = [(v, i) for (v, i) in zip(vjps, inputs) if i.need_grad]
        result = Tensor(result, need_grad=len(children) > 0)
        graph[id(result)].extend(children)
        return result

    return wrapper_operator


def backward(node: Tensor) -> None:
    assert oned(node.value) and node.value.shape[
        0] == 1, f"Sorry this only supports backwards on scalar values. Given shape {node.value.shape}"
    queue: List[Tuple[Tensor, Vjp, Tensor]] = [
        (node, vjp, child) for (vjp, child) in graph[id(node)]
    ]  # (parent, vjp, child)
    while len(queue) != 0:
        parent, vjp, child = queue.pop(0)
        if child.grad is not None:
            print("YAYYYYY")
        child.grad = vjp(
            parent.grad) if child.grad is None else child.grad + vjp(
                parent.grad)

        queue += [(child, vjp, grandkids)
                  for (vjp, grandkids) in graph[id(child)]]


np.random.seed(42)
t = Tensor(np.random.rand(10))
yhat = Tensor(np.random.rand(10), need_grad=False)
print(t.value.shape)
cos = op_wrapper(cos)
mse = op_wrapper(mse)
y = cos(t)
l = mse(y, yhat)
print(l.value.shape)
print(y)
backward(l)
print(t.grad)
# print(y.grad)

import jax.numpy as jnp
from jax import grad

grad_fn = grad(lambda x: ((yhat.value - jnp.cos(x))**2).mean())
print(grad_fn(t.value))

print("LINEAR layer")
W = Tensor(np.random.rand(5, 10))
b = Tensor(np.random.rand(5))
yhat = Tensor(np.random.rand(5), need_grad=False)
graph = defaultdict(list)
linear = op_wrapper(linear)
y = linear(W, b, t)
# y = relu(y)
mse(y, yhat)
l = mse(y, yhat)
backward(l)
print("W grad")
print(W.grad)
grad_fn = grad(lambda W1: ((yhat.value - (W1 @ t.value + b.value))**2).mean())
print("jax grad")
print(grad_fn(W.value))
# testing multiple paths
# i.e grad wrt. W when y = W((Wx + b1) + b2)
print("Multi test")
graph = defaultdict(list)
W = Tensor(np.random.rand(10, 10))
b1 = Tensor(np.random.rand(10))
b2 = Tensor(np.random.rand(10))
t = Tensor(np.random.rand(10), need_grad=False)
yhat = Tensor(np.random.rand(10), need_grad=False)

y1 = linear(W, b1, t)
y2 = linear(W, b2, y1)
l = mse(y2, yhat)
backward(l)
print(W.grad)
print(len(graph.keys()))
print(graph.keys())

gph = {
    id(val): (name, val)
    for (name,
         val) in [("W", W), ("b1",
                             b1), ("b2",
                                   b2), ("t", t), ("yhat",
                                                   yhat), ("y1",
                                                           y1), ("y2",
                                                                 y2), ("l", l)]
}
for i, neigh in graph.items():
    neigh_names = []
    for (v, t) in neigh:
        neigh_names.append(gph[id(t)][0])
    print(f"{gph[i][0]} {neigh_names}")

#     print(f"{x=} {id(x) in graph}")
grad_fn = grad(lambda W: (
    (yhat.value - (W @ (W @ t.value + b1.value) + b2.value))**2).mean())
print(grad_fn(W.value))
import pdb

# pdb.set_trace()
