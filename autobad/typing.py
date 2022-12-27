from typing import Callable, Sequence, Tuple

import numpy as np

Vjp = Callable[[np.array], np.array]
OperatorType = Callable[[Sequence[np.array]], Tuple[np.array, Sequence[Vjp]]]
