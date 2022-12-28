from typing import List, Optional

import numpy as np
from autobad.core import Tensor


def sgd(parameters: List[Tensor],
        lr: float,
        average_gradients: bool = False,
        batch_size: Optional[int] = None) -> None:
    for p in parameters:
        p.value = p.value - (lr * p.grad / batch_size)
        p.grad = None
