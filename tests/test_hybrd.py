"""_summary_
"""
# pylint: disable = W0613, C0116, E1136
import numpy as np
import pytest
from numba import cfunc
from numpy.testing import assert_allclose
from scipy.optimize import fsolve

from cminpack_numba import hybrd1, hybrd_sig

try:
    import NumbaMinpack
except ImportError:
    NumbaMinpack = None


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
def func(x):
    return [x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5]


@cfunc(hybrd_sig)
def func_numba(udata, n, x, fvec, iflag):
    fvec[0] = x[0] * np.cos(x[1]) - 4
    fvec[1] = x[1] * x[0] - x[1] - 5
    return 0


def test_hybrd1_scipy_fsolve():
    scipy_result = fsolve(func, np.array([1.0, 1.0]))
    assert_allclose(scipy_result, np.array([6.50409711, 0.90841421]))
    hybrd1_result = hybrd1(func_numba.address, np.array([1.0, 1.0]))
    assert_allclose(scipy_result, hybrd1_result[0])


@pytest.mark.benchmark(group="hybrd1")
def test_hybrd1_scipy_fsolve_benchmark(benchmark):
    benchmark(fsolve, func, np.array([1.0, 1.0]))


@pytest.mark.benchmark(group="hybrd1")
def test_hybrd1_scipy_fsolve_numba_benchmark(benchmark):
    scipy_result = fsolve(func, np.array([1.0, 1.0]))
    hybrd1_result = hybrd1(func_numba.address, np.array([1.0, 1.0]))
    assert_allclose(scipy_result, hybrd1_result[0])
    benchmark(hybrd1, func_numba.address, np.array([1.0, 1.0]))


@pytest.mark.skipif(NumbaMinpack is None, reason="NumbaMinpack not installed")
@pytest.mark.benchmark(group="hybrd1")
def test_hybrd1_numbaminpack(benchmark):
    @cfunc(NumbaMinpack.minpack_sig)
    def func_numbaminpack(x, fvec, args):
        fvec[0] = x[0] * np.cos(x[1]) - 4
        fvec[1] = x[1] * x[0] - x[1] - 5

    scipy_result = fsolve(func, np.array([1.0, 1.0]))
    hybrd1_result = NumbaMinpack.hybrd(func_numbaminpack.address, np.array([1.0, 1.0]))
    assert_allclose(scipy_result, hybrd1_result[0])
    benchmark(NumbaMinpack.hybrd, func_numbaminpack.address, np.array([1.0, 1.0]))
