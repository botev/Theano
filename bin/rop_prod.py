import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.gradient import zero_grad


def th_normal(shape, mean=0, std=1, dtype=None):
    dtype = dtype or theano.config.floatX
    srng = RandomStreams(np.random.randint(1, 2147462579))
    samples = srng.normal(shape, dtype=dtype)
    samples = zero_grad(samples)
    if std != 1:
        samples *= std
    if mean != 0:
        samples += mean
    return samples


def float64_error():
    a = T.fmatrix()
    b = T.fmatrix()
    shape = (a.shape[0], 30, a.shape[1])
    epsilon = th_normal(shape)
    # epsilon = T.zeros(shape)
    out = a.dimshuffle(0, 'x', 1) * epsilon + a.dimshuffle(0, 'x', 1)
    rop = T.Rop(out, a, b)
    f = theano.function([a, b], rop)
    a_in = np.random.randn(5, 6).astype(theano.config.floatX)
    b_in = np.random.randn(5, 6).astype(theano.config.floatX)
    print(f(a_in, b_in))


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


def verify_rop(fun, pt, n_tests=2, rng=None,
               abs_tol=None, rel_tol=None,
               mode=None,
               cast_to_output_type=False,
               no_debug_ref=True):
    """
    For verifying the Rop we used the following fact:
    We have a function f(w): R^n -> R^d with a Jacobian J of size d x n
    The Lop computes u^T J(w), u is in R^d
    The Rop computes J(w) v, v is in R^n
    Assuming that we have a working Lop we can calculate F = u^T J(w) v
        1. Let ru = u^T J(w) = Lop(f, w, u), then Fu = ru^T v
        2. Let rv = J(w) v = Rop(f, w, v), then Fv = u^T rv
        3. Check that ru and rv are numerically close

    :param fun: a Python function that takes Theano variables as inputs,
        and returns a Theano variable. For instance, an Op instance with
        a single output.
    :param pt: the list of numpy.ndarrays to use as input values.
        These arrays must be either float16, float32, or float64 arrays.
    :param n_tests: number of times to run the test
    :param rng: random number generator used to sample u and v
    :param eps: stepsize used in the Finite Difference Method (Default
        None is type-dependent)
        Raising the value of eps can raise or lower the absolute and
        relative errors of the verification depending on the
        Op. Raising eps does not lower the verification quality
        for linear operations. It
        is better to raise eps than raising abs_tol or rel_tol.
    :param out_type: dtype of output, if complex (i.e. 'complex32' or
        'complex64')
    :param abs_tol: absolute tolerance used as threshold for gradient
        comparison
    :param rel_tol: relative tolerance used as threshold for gradient
        comparison
    :param cast_to_output_type: if the output is float32 and
        cast_to_output_type is True, cast the random projection to
        float32. Otherwise it is float64. float16 is not handled here.
    :param no_debug_ref: Don't use DebugMode for the numerical
        gradient function.

    :note: This function does not support multiple outputs. In
        tests/test_scan.py there is an experimental verify_grad that
        covers that case as well by using random projections.
    """
    import six.moves.builtins as builtins
    from theano.tests.unittest_tools import seed_rng
    from theano.gradient import mode_not_slow, Lop, Rop, numeric_grad

    pt = [np.array(p) for p in pt]

    for i, p in enumerate(pt):
        if p.dtype not in ('float16', 'float32', 'float64'):
            raise TypeError(
                ('verify_grad can work only with floating point '
                 'inputs, but input %i has dtype "%s".') % (i, p.dtype))

    if rng is None:
        seed_rng()
        rng = np.random
        # The import is here to prevent circular import.
        from theano import compile, shared
        import theano.tensor
        from theano.tensor import as_tensor_variable, TensorType
        for i, p in enumerate(pt):
            if p.dtype not in ('float16', 'float32', 'float64'):
                raise TypeError(
                    ('verify_grad can work only with floating point '
                     'inputs, but input %i has dtype "%s".') % (i, p.dtype))

        _type_tol = dict(  # relative error tolerances for different types
            float16=1e-3,
            float32=1e-6,
            float64=1e-12)

        if abs_tol is None:
            abs_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)
        if rel_tol is None:
            rel_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)

        if rng is None:
            raise TypeError(('rng should be a valid instance of '
                             'numpy.random.RandomState. You may '
                             'want to use theano.tests.unittest'
                             '_tools.verify_grad instead of '
                             'theano.gradient.verify_grad.'))

        # We allow input downcast in function, because numeric_grad works in the
        # most precise dtype used among the inputs, so we may need to cast some.
        def function(inputs, output, name, mode=mode):
            f = compile.function(inputs, output, accept_inplace=True,
                                 allow_input_downcast=True, mode=mode,
                                 on_unused_input='ignore', name=name)
            return f

        tensor_pt = [
            TensorType(
                as_tensor_variable(p).dtype,
                as_tensor_variable(p).broadcastable)(name='input %i' % i)
            for i, p in enumerate(pt)]

        # fun can be either a function or an actual Op instance
        o_output = fun(*tensor_pt)

        if isinstance(o_output, list):
            raise NotImplementedError(('cant (yet) autotest Rop of fun '
                                       'with multiple outputs'))
            # we could make loop over outputs making random projections R for each,
            # but this doesn't handle the case where not all the outputs are
            # differentiable... so I leave this as TODO for now -JB.

        o_fn = function([tensor_pt], o_output, name='gradient.py fwd')
        o_fn_out = o_fn(*pt)

        if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
            raise TypeError(
                'It seems like you are trying to use verify_grad '
                'on an op or a function which outputs a list: there should'
                ' be a single (array-like) output instead')

        # random_projection should not have elements too small,
        # otherwise too much precision is lost in numerical gradient
        def random_projection(array):
            plain = rng.rand(*array.shape) + 0.5
            if cast_to_output_type and array.dtype == "float32":
                return np.array(plain, array.dtype)
            return plain

        # From here in we are running under the assumption that there is a single pt
        # Create the two projection variables
        u = shared(random_projection(o_fn_out), borrow=True)
        u.name = 'random_projection_u'
        v = [shared(random_projection(p), borrow=True) for p in pt]
        for i, v_shared in enumerate(v):
            v_shared.name = 'random_projection_v_%d' % i

        # Compute Fu and Fv from the function doc
        uJ = Lop(o_output, tensor_pt, u, disconnected_inputs='ignore')
        Fu = sum(theano.tensor.sum(uJi * vi) for uJi, vi in zip(uJ, v))
        Jv = Rop(o_output, tensor_pt, v)
        Fv = theano.tensor.sum(u * Jv)

        if no_debug_ref:
            mode_for_cost = mode_not_slow(mode)
        else:
            mode_for_cost = mode

        # Compile the function to calculate them
        calc_fn = function(tensor_pt, [Fu, Fv], name="difference.py",
                           mode=mode_for_cost)

        # Run the tests
        for _ in range(n_tests):
            fu, fv = calc_fn(*pt)
            print(fu-fv)
            abs_err, rel_err = numeric_grad.abs_rel_err(fu, fv)
            if abs_err > abs_tol or rel_err > rel_tol:
                err = verify_rop.E_grad(0, 0, (), fu, fv,
                                        abs_err, rel_err,
                                        abs_tol, rel_tol)
                err.args += ("\nThe error happened with the following inputs:", pt)
                raise err
            # Set u and v for the next test
            u.set_value(random_projection(o_fn_out))
            for v_shared, p in zip(v, pt):
                v_shared.set_value(random_projection(p))


if __name__ == '__main__':
    def prod_fn(a):
        return T.prod(a, axis=1)
    a_pt = np.random.randn(20, 20).astype(theano.config.floatX)
    # Without zeros
    verify_rop(prod_fn, [a_pt])
    # With one zero
    a_pt[:, 0] = 0
    verify_rop(prod_fn, [a_pt])
    # With two zeros
    a_pt[:, 1] = 0
    verify_rop(prod_fn, [a_pt])
    # float64_error()
    # f = make_function()
    # for _ in range(5):
    #     a = np.random.randn(15, 5).astype(theano.config.floatX) * 10
    #     b = np.random.randn(15, 5).astype(theano.config.floatX) * 10
    #     general_test(a, b, f)
    #     multiple_zeros(a, b, f)
    #     one_zero_test(a, b, f)