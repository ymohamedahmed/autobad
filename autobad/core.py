from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict, List, Sequence
from collections import defaultdict
from autobad.typing import Vjp
import numpy as np
from dataclasses import dataclass
from typing import Optional
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


def backward(node: Tensor) -> None:
    from autobad.graph import Graph
    oned = lambda v: len(v.shape) == 1
    assert oned(node.value) and node.value.shape[
        0] == 1, f"Sorry this only supports backwards on scalar values. Given shape {node.value.shape}"
    queue: List[Tuple[Tensor, Vjp,
                      Tensor]] = [(node, vjp, child)
                                  for (vjp, child) in Graph.get(node)]
    # (parent, vjp, child)
    while len(queue) != 0:
        parent, vjp, child = queue.pop(0)
        downstream = parent.grad
        child.grad = vjp(
            downstream) if child.grad is None else child.grad + vjp(downstream)

        queue += [(child, vjp, grandkids)
                  for (vjp, grandkids) in Graph.get(child)]
    Graph.clear()
