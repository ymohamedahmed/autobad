import numpy as np
from autobad.ops import linear, mse, cos, relu, softmax, cross_entropy
from autobad.core import Tensor, backward
from autobad.graph import Graph
import jax
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


def test_multiple_backwards():
    test_cos()
    test_reused_parameter()


def test_relu():
    x = Tensor(np.random.rand(10))
    W = Tensor(np.random.rand(10, 10))
    b = Tensor(np.random.rand(10))

    yhat = Tensor(np.random.rand(10), need_grad=False)
    y1 = linear(W, b, x)
    y2 = relu(y1)
    l = mse(y2, yhat)
    backward(l)
    value, grads = value_and_grad(lambda W, b, x: (
        (yhat.value - jax.nn.relu(W @ x + b))**2).mean(),
                                  argnums=[0, 1, 2])(W.value, b.value, x.value)
    assert np.allclose(l.value, value)
    assert np.allclose(W.grad, grads[0])
    assert np.allclose(b.grad, grads[1])
    assert np.allclose(x.grad, grads[2])


def test_softmax():
    x = Tensor(np.random.rand(10))
    W = Tensor(np.random.rand(10, 10))
    b = Tensor(np.random.rand(10))

    yhat = Tensor(np.random.rand(10), need_grad=False)
    y1 = linear(W, b, x)
    y2 = softmax(y1)
    l = mse(y2, yhat)
    backward(l)
    value, grads = value_and_grad(lambda W, b, x: (
        (yhat.value - jax.nn.softmax(W @ x + b))**2).mean(),
                                  argnums=[0, 1, 2])(W.value, b.value, x.value)
    assert np.allclose(l.value, value)
    assert np.allclose(W.grad, grads[0])
    assert np.allclose(b.grad, grads[1])
    assert np.allclose(x.grad, grads[2])


def test_cross_entropy():
    import torch
    import torch.nn.functional as F
    x = Tensor(np.random.rand(10))
    W = Tensor(np.random.rand(10, 10))
    b = Tensor(np.random.rand(10))

    y_true = np.random.rand(10)
    y_true = y_true / y_true.sum()
    y_true = Tensor(y_true, need_grad=False)

    y1 = linear(W, b, x)
    y2 = softmax(y1)
    l = cross_entropy(y_true, y2)
    # l = cross_entropy(y2, y_true)
    backward(l)
    value, grads = value_and_grad(
        lambda W, b, x:
        (-y_true.value * jnp.log(jax.nn.softmax(W @ x + b))).sum(),
        argnums=[0, 1, 2])(W.value, b.value, x.value)

    # use y1 here since torch takes softmax
    torch_answer = F.cross_entropy(torch.Tensor(y1.value),
                                   torch.Tensor(y_true.value))

    print(value)
    assert np.allclose(l.value, torch_answer)
    assert np.allclose(l.value, value)
    assert np.allclose(W.grad, grads[0])
    assert np.allclose(b.grad, grads[1])
    assert np.allclose(x.grad, grads[2])


def test_need_grad():
    x = Tensor(np.random.rand(10), need_grad=False)
    W = Tensor(np.random.rand(10, 10), need_grad=False)
    b = Tensor(np.random.rand(10), need_grad=False)

    yhat = Tensor(np.random.rand(10), need_grad=False)
    y1 = linear(W, b, x)
    y2 = softmax(y1)
    l = mse(y2, yhat)

    assert len(Graph.get_instance()._graph.keys()) == 0