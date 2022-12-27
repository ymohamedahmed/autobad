from core import Tensor
from typing import List


def sgd(parameters: List[Tensor], lr: float) -> None:
    for p in parameters:
        p.value -= lr * p.grad
        p.grad = None