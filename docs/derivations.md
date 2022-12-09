# Linear layer

Define $y = Wx + b$ 

Given a downstream gradient, $g$, i.e. something like $\delta l / \delta y_n \cdot \delta y_n/\delta y_{n-1}$

# MSE 

Vectorized mse is  $$m = 1/N*(x-y)^T(x-y) = 1/N * (x^Tx + y^Ty - 2x^Ty)$$

Recall that $x^Ty = y^Tx$
and therefore $dm/dx = 1/N(2x - 2y)$ and $dm/dy = 1/N(2y-2x)$

# Softmax

Denote softmax as $s$, recall it is defined as 

$s(\mathbb{x})_i = \frac{ e^{x_i}}{\sum_k e^{x_k}}$

Let's compute the Jacobian.

From the quotient rule
$$\frac{\delta s(x)_i}{\delta x_j} = \frac{\frac{\delta e^{x_i}}{\delta  x_j}(\sum_k e^{x_k})-e^{x_i}\frac{\delta \sum_k e^{x_k}}{ \delta x_j}}{(\sum_k e^{x_k})^2} = \frac{\frac{\delta e^{x_i}}{\delta  x_j}(\sum_k e^{x_k})-e^{x_i}e^{x_j}}{(\sum_k e^{x_k})^2}$$
Denote $S=\sum_k e^{x_k}$, 
$$\frac{\delta s(x)_i}{\delta x_j} = \frac{\frac{\delta e^{x_i}}{\delta  x_j}S-e^{x_i}e^{x_j}}{S^2}$$
If $i=j$,
$$\frac{\delta s(x)_i}{\delta x_i} = \frac{e^{x_i}S-e^{x_i}e^{x_i}}{S^2} = \frac{e^{x_i}(S-e^{x_i})}{S^2}  = \frac{e^{x_i}}{S}\frac{(S-e^{x_i})}{S}=s(\mathbb{x})_i(1-s(\mathbb{x})_i)$$

If $i\neq j$, 
$$\frac{\delta s(x)_i}{\delta x_j} = \frac{-e^{x_i}e^{x_j}}{S^2}=-s(\mathbb{x})_is(\mathbb{x})_j$$

Hence the computation in `ops.py`.

<!-- # cross entropy

$ -->