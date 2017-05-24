import numpy as np
import theano
import theano.tensor as T


def make_function():
    a_in = T.fmatrix()
    a_eval = T.fmatrix()
    prod = T.prod(a_in, axis=1)
    out = T.Rop(prod, a_in, a_eval)
    return theano.function([a_in, a_eval], out)


def general_test(a_in, a_eval, f):
    assert a_in.ndim == 2
    assert a_in.shape == a_eval.shape
    rop_th = f(a_in, a_eval)
    a_out = np.prod(a_in, axis=1)
    rop = np.sum(a_out[:, np.newaxis] * (a_eval / a_in), axis=1)
    assert rop.shape == rop_th.shape
    assert np.allclose(rop, rop_th)


def one_zero_test(a_in, a_eval, f):
    assert a_in.ndim == 2
    assert a_in.shape == a_eval.shape
    ind = np.random.randint(0, high=a_in.shape[1], size=(a_in.shape[0],))
    i1 = np.arange(a_in.shape[0])
    a_in[i1, ind] = 0
    rop_th = f(a_in, a_eval)
    a_in[i1, ind] = 1
    a_out = np.prod(a_in, axis=1)
    rop = a_out * a_eval[i1, ind]
    assert rop.shape == rop_th.shape
    assert np.allclose(rop, rop_th)


def multiple_zeros(a_in, a_eval, f):
    assert a_in.ndim == 2
    assert a_in.shape == a_eval.shape
    a_in = np.concatenate((a_in, np.zeros((a_in.shape[0], 2), dtype=a_in.dtype)), axis=1)
    a_eval = np.concatenate((a_eval, np.zeros((a_in.shape[0], 2), dtype=a_in.dtype)), axis=1)
    rop_th = f(a_in, a_eval)
    rop = np.zeros((a_in.shape[0], ), dtype=a_in.dtype)
    assert rop.shape == rop_th.shape
    assert np.allclose(rop_th, rop)


if __name__ == '__main__':
    f = make_function()
    for _ in range(5):
        a = np.random.randn(15, 5).astype(theano.config.floatX) * 10
        b = np.random.randn(15, 5).astype(theano.config.floatX) * 10
        general_test(a, b, f)
        multiple_zeros(a, b, f)
        one_zero_test(a, b, f)

