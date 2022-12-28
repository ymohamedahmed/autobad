from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np

from autobad.typing import Vjp
"""
Simple implementation of eager, reverse-mode and vectorized autodiff. Only dependency is numpy.

No clever handling of batched inputs.

"""


@dataclass
class Tensor:
    value: np.array
    need_grad: Optional[bool] = True
    grad: Optional[np.array] = None
    _count: Optional[int] = 0


def backward(node: Tensor) -> None:
    from autobad.graph import Graph

    oned = lambda v: len(v.shape) == 1
    assert (
        oned(node.value) and node.value.shape[0] == 1
    ), f"Sorry this only supports backwards on scalar values. Given shape {node.value.shape}"
    queue: Queue[Tuple[Tensor, Vjp, Tensor]] = Queue()

    many_put = lambda children: list(map(queue.put, children))
    many_put([(node, vjp, child) for (vjp, child) in Graph.get(node)])
    while not queue.empty():
        parent, vjp, child = queue.get()
        downstream = parent.grad
        child._count += 1
        child.grad = vjp(
            downstream) if child.grad is None else child.grad + vjp(downstream)
        # child.grad = vjp(downstream) if child.grad is None else (
        #     child.grad * (child._count - 1)) + (vjp(downstream) /
        #                                         (child._count + 1))
        many_put([(child, vjp, grandkids)
                  for (vjp, grandkids) in Graph.get(child)])
    Graph.clear()
