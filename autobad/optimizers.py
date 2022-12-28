from typing import List

from autobad.core import Tensor


def sgd(parameters: List[Tensor], lr: float) -> None:
    for p in parameters:
        p.value = p.value - (lr * p.grad)
        p.grad = None
