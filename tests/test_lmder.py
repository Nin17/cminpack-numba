"""_summary_
"""

# pylint: disable = W0613, C0116, E1136

import numpy as np
import pytest
from numba import carray, cfunc, farray
from numpy.testing import assert_allclose
from scipy.optimize import least_squares, leastsq

from cminpack_numba import lmder1, lmder_sig


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
def func(x):
    return 2 * (x - 3) ** 2 + 1


def Dfun(x):
    return 4 * (x - 3)


@cfunc(lmder_sig)
def func_numba(udata, m, n, x, fvec, fjac, ldfjac, iflag):
    if iflag == 1:
        fvec[0] = 2 * (x[0] - 3) ** 2 + 1
    if iflag == 2:
        fjac = farray(fjac, (1, 1), np.float64)
        fjac[0, 0] = 4 * (x[0] - 3)
    return 0


@cfunc(lmder_sig)
def func_numba_udata(udata, m, n, x, fvec, fjac, ldfjac, iflag):
    udata = carray(udata, (3,), np.float64)
    if iflag == 1:
        fvec[0] = udata[0] * (x[0] - udata[1]) ** 2 + udata[2]
    if iflag == 2:
        fjac = farray(fjac, (1, 1), np.float64)
        fjac[0, 0] = 2 * udata[0] * (x[0] - udata[1])
    return 0


# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
def rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def jac_rosenbrock(x):
    return np.array([[-20 * x[0], 10], [-1, 0]])


@cfunc(lmder_sig)
def rosenbrock_numba(udata, m, n, x, fvec, fjac, ldfjac, iflag):
    if iflag == 1:
        fvec[0] = 10.0 * (x[1] - x[0] ** 2)
        fvec[1] = 1.0 - x[0]
    if iflag == 2:
        fjac = farray(fjac, (2, 2), np.float64)
        fjac[0, 0] = -20 * x[0]
        fjac[0, 1] = 10.0
        fjac[1, 0] = -1.0
        fjac[1, 1] = 0
    return 0


@cfunc(lmder_sig)
def rosenbrock_numba_udata(udata, m, n, x, fvec, fjac, ldfjac, iflag):
    udata = carray(udata, (2,), np.float64)
    if iflag == 1:
        fvec[0] = udata[0] * (x[1] - x[0] ** 2)
        fvec[1] = udata[1] - x[0]
    if iflag == 2:
        fjac = farray(fjac, (2, 2), np.float64)
        fjac[0, 0] = -2 * udata[0] * x[0]
        fjac[0, 1] = udata[0]
        fjac[1, 0] = -1.0
        fjac[1, 1] = 0.0
    return 0


def test_lmder1_scipy_leastsq():
    scipy_result = leastsq(func, 0, Dfun=Dfun)

    assert_allclose(scipy_result[0], np.array([2.99999999]))
    assert scipy_result[1] == 1
    lmder1_result = lmder1(func_numba.address, 1, np.array([0.0]))
    assert_allclose(scipy_result[0], lmder1_result[0])


def test_lmder1_scipy_least_squares():
    x0_rosenbrock = np.array([2, 2])
    scipy_result = least_squares(rosenbrock, x0_rosenbrock, jac=jac_rosenbrock)
    assert_allclose(scipy_result.x, np.array([1.0, 1.0]))

    lmder1_result = lmder1(rosenbrock_numba.address, 2, np.array([2.0, 2.0]))
    assert_allclose(scipy_result.x, lmder1_result[0])


@pytest.mark.benchmark(group="lmder1")
def test_leastsq_func(benchmark):
    benchmark(leastsq, func, 0, Dfun=Dfun)


@pytest.mark.benchmark(group="lmder1")
def test_leastsq_rosenbrock(benchmark):
    benchmark(leastsq, rosenbrock, np.array([2.0, 2.0]), Dfun=jac_rosenbrock)


@pytest.mark.benchmark(group="lmder1")
def test_least_squares_func(benchmark):
    benchmark(least_squares, func, 0, jac=Dfun, method="lm")


@pytest.mark.benchmark(group="lmder1")
def test_least_squares_rosenbrock(benchmark):
    benchmark(
        least_squares, rosenbrock, np.array([2.0, 2.0]), jac=jac_rosenbrock, method="lm"
    )


@pytest.mark.benchmark(group="lmder1")
def test_lmder1_func(benchmark):
    scipy_result = leastsq(func, 0, Dfun=Dfun)
    lmder1_result = lmder1(func_numba.address, 1, np.array([0.0]))
    assert_allclose(scipy_result[0], lmder1_result[0])
    benchmark(lmder1, func_numba.address, 1, np.array([0.0]))


@pytest.mark.benchmark(group="lmder1")
def test_lmder1_func_udata(benchmark):
    scipy_result = leastsq(func, 0, Dfun=Dfun)
    udata = np.array([2.0, 3.0, 1.0])
    lmder1_result = lmder1(func_numba_udata.address, 1, np.array([0.0]), udata=udata)
    assert_allclose(scipy_result[0], lmder1_result[0])
    benchmark(lmder1, func_numba_udata.address, 1, np.array([0.0]), udata=udata)


@pytest.mark.benchmark(group="lmder1")
def test_lmder1_rosenbrock(benchmark):
    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]), jac=jac_rosenbrock)
    lmder1_result = lmder1(rosenbrock_numba.address, 2, np.array([2.0, 2.0]))
    assert_allclose(scipy_result.x, lmder1_result[0])
    benchmark(lmder1, rosenbrock_numba.address, 2, np.array([2.0, 2.0]))


@pytest.mark.benchmark(group="lmder1")
def test_lmder1_rosenbrock_udata(benchmark):
    scipy_result = least_squares(rosenbrock, np.array([2.0, 2.0]), jac=jac_rosenbrock)
    udata = np.array([10.0, 1.0])
    lmder1_result = lmder1(
        rosenbrock_numba_udata.address, 2, np.array([2.0, 2.0]), udata=udata
    )
    assert_allclose(scipy_result.x, lmder1_result[0])
    benchmark(
        lmder1, rosenbrock_numba_udata.address, 2, np.array([2.0, 2.0]), udata=udata
    )
