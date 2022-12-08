# Linear layer

Define $y = Wx + b$ 

Given a downstream gradient, $g$, i.e. something like $\delta l / \delta y_n \cdot \delta y_n/\delta y_{n-1}$

# MSE 

Vectorized mse is  $$m = 1/N*(x-y)^T(x-y) = 1/N * (x^Tx + y^Ty - 2x^Ty)$$

Recall that $x^Ty = y^Tx$
and therefore $dm/dx = 1/N(2x - 2y)$ and $dm/dy = 1/N(2y-2x)$