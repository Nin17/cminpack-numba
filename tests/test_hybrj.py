"""_summary_
"""
# pylint: disable = W0613, C0116, E1136, R0913
import numpy as np
from numba import cfunc, farray
from numpy.testing import assert_allclose
from scipy.optimize import fsolve

from cminpack_numba import hybrj1, hybrj_sig

import pytest


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
def func(x):
    return [x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5]

def fprime(x):
    return [[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]]


@cfunc(hybrj_sig)
def func_numba(udata, n, x, fvec, fjac, ldfjac, iflag):
    if iflag == 1:
        fvec[0] = x[0] * np.cos(x[1]) - 4
        fvec[1] = x[1] * x[0] - x[1] - 5
    elif iflag == 2:
        fjac = farray(fjac, (2, 2), np.float64)
        fjac[0, 0] = np.cos(x[1])
        fjac[0, 1] = -x[0] * np.sin(x[1])
        fjac[1, 0] = x[1]
        fjac[1, 1] = x[0] - 1
    return 0


def test_hybrj1_scipy_fsolve():
    scipy_result = fsolve(func, np.array([1.0, 1.0]), fprime=fprime)
    assert_allclose(scipy_result, np.array([6.50409711, 0.90841421]))
    hybrj1_result = hybrj1(func_numba.address, np.array([1.0, 1.0]))
    assert_allclose(scipy_result, hybrj1_result[0])


@pytest.mark.benchmark(group="hybrj1")
def test_hybrj1_scipy_fsove_benchmark(benchmark):
    benchmark(fsolve, func, np.array([1.0, 1.0]), fprime=fprime)
