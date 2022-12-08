import numpy as np
from autobad.core import linear, mse, cos, Tensor, backward
from jax import value_and_grad
import jax.numpy as jnp


def test_cos():
    x = Tensor(np.random.rand(10))
    yhat = Tensor(np.random.rand(10), need_grad=False)
    y = cos(x)
    l = mse(y, yhat)
    backward(l)
    value, grad = value_and_grad(
        lambda x1: ((yhat.value - jnp.cos(x1))**2).mean())(x.value)
    assert np.allclose(l.value, value)
    assert np.allclose(x.grad, grad)


def test_reused_parameter():
    """
    Tests the case where the same parameter is used in multiple layers.
    (i.e. gradient must be summed)
    Also tests linear layer implementation.
    """
    W = Tensor(np.random.rand(10, 10))
    b1 = Tensor(np.random.rand(10))
    b2 = Tensor(np.random.rand(10))
    t = Tensor(np.random.rand(10), need_grad=False)
    yhat = Tensor(np.random.rand(10), need_grad=False)

    y1 = linear(W, b1, t)
    y2 = linear(W, b2, y1)
    l = mse(y2, yhat)
    backward(l)

    value, grads = value_and_grad(lambda W, b1, b2: (
        (yhat.value.copy() - (W @ (W @ t.value + b1) + b2))**2).mean(),
                                  argnums=[0, 1, 2])(W.value, b1.value,
                                                     b2.value)
    assert np.allclose(W.grad, grads[0])
    assert np.allclose(b1.grad, grads[1])
    assert np.allclose(b2.grad, grads[2])
    assert np.allclose(l.value, value)
