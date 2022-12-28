from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple

import numpy as np

from autobad.typing import Vjp


@dataclass
class Tensor:
    value: np.array
    need_grad: Optional[bool] = True
    grad: Optional[np.array] = None


def backward(node: Tensor) -> None:
    from autobad.graph import Graph

    oned = lambda v: len(v.shape) == 1
    assert (
        oned(node.value) and node.value.shape[0] == 1
    ), f"Sorry this only supports backwards on scalar values. Given shape {node.value.shape}"

    # Queue is used for breadth-first traversal, each element is
    # a tuple of (parent tensor, vector jacobian product, child tensor).
    queue: Queue[Tuple[Tensor, Vjp, Tensor]] = Queue()

    many_put = lambda children: list(map(queue.put, children))
    many_put([(node, vjp, child) for (vjp, child) in Graph.get(node)])
    while not queue.empty():
        parent, vjp, child = queue.get()
        downstream = parent.grad
        child.grad = (
            vjp(downstream) if child.grad is None else child.grad + vjp(downstream)
        )
        many_put([(child, vjp, grandkids) for (vjp, grandkids) in Graph.get(child)])
    Graph.clear()
