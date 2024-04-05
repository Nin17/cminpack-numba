"""_summary_
"""

import numpy as np
import pytest
from numba import carray, cfunc
from numpy.testing import assert_allclose
from scipy.optimize import least_squares, leastsq

from cminpack_numba import lmdif1, lmdif_sig

try:
    import NumbaMinpack
except ImportError:
    NumbaMinpack = None


# --------------------------------------- func --------------------------------------- #


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
def func(x):
    return 2.0 * (x - 3.0) ** 2 + 1.0


def func_args(x, *args):
    return args[0] * (x - args[1]) ** 2 + args[2]


scipy_result_func = leastsq(func, 0)
assert_allclose(scipy_result_func[0], np.array([2.99999999]))
assert scipy_result_func[1] == 1


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_leastsq(benchmark):
    benchmark(leastsq, func, 0)


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_args_leastsq(benchmark):
    benchmark(leastsq, func_args, 0, args=(2.0, 3.0, 1.0))


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_least_squares(benchmark):
    benchmark(least_squares, func, 0, method="lm")


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_args_least_squares(benchmark):
    benchmark(least_squares, func_args, 0, method="lm", args=(2.0, 3.0, 1.0))


@cfunc(lmdif_sig)
def func_numba(udata, m, n, x, fvec, iflag):
    fvec[0] = 2.0 * (x[0] - 3.0) ** 2 + 1.0
    return 0


@cfunc(lmdif_sig)
def func_numba_udata(udata, m, n, x, fvec, iflag):
    udata = carray(udata, (3,), np.float64)
    fvec[0] = udata[0] * (x[0] - udata[1]) ** 2 + udata[2]
    return 0


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_lmdif1(benchmark):
    lmdif1_result = lmdif1(func_numba.address, 1, np.array([0.0]))
    assert_allclose(scipy_result_func[0], lmdif1_result[0])
    benchmark(lmdif1, func_numba.address, 1, np.array([0.0]))


@pytest.mark.benchmark(group="lmdif1_func")
def test_func_lmdif1_udata(benchmark):
    udata = np.array([2.0, 3.0, 1.0])
    lmdif1_result = lmdif1(func_numba_udata.address, 1, np.array([0.0]), udata=udata)
    assert_allclose(scipy_result_func[0], lmdif1_result[0])
    benchmark(lmdif1, func_numba_udata.address, 1, np.array([0.0]), udata=udata)


@pytest.mark.skipif(NumbaMinpack is None, reason="NumbaMinpack not installed")
@pytest.mark.benchmark(group="lmdif1_func")
def test_func_NumbaMinpack(benchmark):
    @cfunc(NumbaMinpack.minpack_sig)
    def func_numbaminpack(x, fvec, args):
        fvec[0] = 2.0 * (x[0] - 3.0) ** 2 + 1.0

    scipy_result = leastsq(func, 0)
    lmdif1_result = NumbaMinpack.lmdif(func_numbaminpack.address, np.array([0.0]), 1)
    assert_allclose(scipy_result[0], lmdif1_result[0])
    benchmark(NumbaMinpack.lmdif, func_numbaminpack.address, np.array([0.0]), 1)


@pytest.mark.skipif(NumbaMinpack is None, reason="NumbaMinpack not installed")
@pytest.mark.benchmark(group="lmdif1_func")
def test_func_NumbaMinpack_udata(benchmark):
    @cfunc(NumbaMinpack.minpack_sig)
    def func_numbaminpack(x, fvec, args):
        fvec[0] = args[0] * (x[0] - args[1]) ** 2 + args[2]

    scipy_result = leastsq(func, 0)
    udata = np.array([2.0, 3.0, 1.0])
    lmdif1_result = NumbaMinpack.lmdif(
        func_numbaminpack.address, np.array([0.0]), 1, args=udata
    )
    assert_allclose(scipy_result[0], lmdif1_result[0])
    benchmark(
        NumbaMinpack.lmdif, func_numbaminpack.address, np.array([0.0]), 1, args=udata
    )


# ------------------------------------ rosenbrock ------------------------------------ #


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
def rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


@cfunc(lmdif_sig)
def rosenbrock_numba(udata, m, n, x, fvec, iflag):
    fvec[0] = 10.0 * (x[1] - x[0] ** 2)
    fvec[1] = 1.0 - x[0]
    return 0


@cfunc(lmdif_sig)
def rosenbrock_numba_udata(udata, m, n, x, fvec, iflag):
    udata = carray(udata, (2,), np.float64)
    fvec[0] = udata[0] * (x[1] - x[0] ** 2)
    fvec[1] = udata[1] - x[0]
    return 0


def test_lmdif1_scipy_least_squares():
    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]))
    assert_allclose(scipy_result.x, np.array([1.0, 1.0]))
    lmdif1_result = lmdif1(rosenbrock_numba.address, 2, np.array([2.0, 2.0]))
    assert_allclose(scipy_result.x, lmdif1_result[0])


@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_leastsq_rosenbrock(benchmark):
    benchmark(leastsq, rosenbrock, np.array([2.0, 2.0]))


@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_least_squares_rosenbrock(benchmark):
    benchmark(least_squares, rosenbrock, np.array([2.0, 2.0]), method="lm")


@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_lmdif1_rosenbrock(benchmark):
    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]))
    lmdif1_result = lmdif1(rosenbrock_numba.address, 2, np.array([2.0, 2.0]))
    assert_allclose(scipy_result.x, lmdif1_result[0])
    benchmark(lmdif1, rosenbrock_numba.address, 2, np.array([2.0, 2.0]))


@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_lmdif1_rosenbrock_udata(benchmark):
    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]))
    udata = np.array([10.0, 1.0])
    lmdif1_result = lmdif1(
        rosenbrock_numba.address, 2, np.array([2.0, 2.0]), udata=udata
    )
    assert_allclose(scipy_result.x, lmdif1_result[0])
    benchmark(
        lmdif1, rosenbrock_numba_udata.address, 2, np.array([2.0, 2.0]), udata=udata
    )


@pytest.mark.skipif(NumbaMinpack is None, reason="NumbaMinpack not installed")
@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_lmdif1_rosenbrock_numbaminpack(benchmark):
    @cfunc(NumbaMinpack.minpack_sig)
    def rosenbrock_numbaminpack(x, fvec, args):
        fvec[0] = 10.0 * (x[1] - x[0] ** 2)
        fvec[1] = 1.0 - x[0]

    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]))
    lmdif1_result = NumbaMinpack.lmdif(
        rosenbrock_numbaminpack.address, np.array([2.0, 2.0]), 2
    )
    assert_allclose(scipy_result.x, lmdif1_result[0])
    benchmark(
        NumbaMinpack.lmdif, rosenbrock_numbaminpack.address, np.array([2.0, 2.0]), 2
    )


@pytest.mark.skipif(NumbaMinpack is None, reason="NumbaMinpack not installed")
@pytest.mark.benchmark(group="lmdif1_rosenbrock")
def test_lmdif1_rosenbrock_udata_numbaminpack(benchmark):
    @cfunc(NumbaMinpack.minpack_sig)
    def rosenbrock_numbaminpack(x, fvec, args):
        fvec[0] = args[0] * (x[1] - x[0] ** 2)
        fvec[1] = args[1] - x[0]

    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]))
    udata = np.array([10.0, 1.0])
    lmdif1_result = NumbaMinpack.lmdif(
        rosenbrock_numbaminpack.address, np.array([2.0, 2.0]), 2, args=udata
    )
    assert_allclose(scipy_result.x, lmdif1_result[0])
    benchmark(
        NumbaMinpack.lmdif,
        rosenbrock_numbaminpack.address,
        np.array([2.0, 2.0]),
        2,
        args=udata,
    )
