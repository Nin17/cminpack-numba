"""Python implementations of the tests from the minpack c-api test suite."""

from __future__ import annotations

from numba import carray, cfunc, njit
from numpy import array, empty, finfo, float64, full, int32, ones, sqrt
from numpy.testing import assert_allclose, assert_equal

from cminpack_numba import enorm, hybrd, hybrd1, hybrd1_, hybrd_, hybrd_sig
from cminpack_numba.utils import ptr_from_val, val_from_ptr

# ruff: noqa: ANN001, ARG001

UDATA = array([3.0, 2.0, 1.0, 0.0])
REFERENCE = array(
    [
        -0.5706545,
        -0.6816283,
        -0.7017325,
        -0.7042129,
        -0.7013690,
        -0.6918656,
        -0.6657920,
        -0.5960342,
        -0.4164121,
    ],
)
TOL = sqrt(finfo(float64).eps)
N = 9
LWA = 180
X0 = full(N, -1.0)
DIAG = ones(N)
WA = empty(LWA)


def _check_result(x, fvec, info, nfev=None, tol=TOL) -> None:
    assert_equal(info, 1)
    if nfev is not None:
        assert_equal(nfev, 14)
    assert_allclose(enorm(fvec), 0.0, atol=tol)
    assert_allclose(x, REFERENCE, atol=10 * tol)


@cfunc(hybrd_sig)
def trial_hybrd_fcn(udata, n, x, fvec, iflag):  # noqa: ANN201, D103
    if iflag == 0:
        return 0
    for k in range(n):
        temp = (3.0 - 2.0 * x[k]) * x[k]
        temp1 = x[k - 1] if k != 0 else 0.0
        temp2 = x[k + 1] if k != n - 1 else 0.0
        fvec[k] = temp - temp1 - 2.0 * temp2 + 1.0
    return 0


@cfunc(hybrd_sig)
def trial_hybrd_fcn_udata(udata, n, x, fvec, iflag):  # noqa: ANN201, D103
    udata = carray(udata, (4,), dtype=float64)
    if iflag == 0:
        return 0
    for k in range(n):
        temp = (udata[0] - udata[1] * x[k]) * x[k]
        temp1 = x[k - 1] if k != 0 else udata[3]
        temp2 = x[k + 1] if k != n - 1 else udata[3]
        fvec[k] = temp - temp1 - udata[1] * temp2 + udata[2]
    return 0


@njit
def driver(address, udata=None):  # noqa: ANN201, D103
    nfevptr = ptr_from_val(int32(0))
    x = X0.copy()
    fvec = empty(N)
    diag = DIAG.copy()
    fjac = empty((N, N))
    lr = N * (N + 1) // 2
    r = empty(lr)
    qtf = empty(N)
    wa1 = empty(N)
    wa2 = empty(N)
    wa3 = empty(N)
    wa4 = empty(N)

    args = address, N, x, fvec, TOL, 2000, 1, 1, 0.0, diag, 2, 100.0, 0
    args2 = nfevptr, fjac, N, r, lr, qtf, wa1, wa2, wa3, wa4, udata
    _, _, _, _, _, _, info = hybrd_(*args, *args2)
    return x, fvec, info, val_from_ptr(nfevptr)


def test_hybrd1() -> None:
    """Python implementation of the minpack c-api hybrd1 test."""
    x, fvec, info = hybrd1(trial_hybrd_fcn.address, X0, TOL)
    _check_result(x, fvec, info)


def test_hybrd1_() -> None:
    """Python implementation of the minpack c-api hybrd1 test."""
    x = X0.copy()
    fvec = empty(N)
    _, _, info = hybrd1_(trial_hybrd_fcn.address, N, x, fvec, TOL, WA, LWA)
    _check_result(x, fvec, info)


def test_hybrd() -> None:
    """Python implementation of the minpack c-api hybrd test."""
    args = trial_hybrd_fcn.address, X0, TOL, 2000, 1, 1, 0.0, DIAG, 2, 100.0, 0
    x, fvec, _, _, _, nfev, info = hybrd(*args)
    _check_result(x, fvec, info, nfev)


def test_hybrd_() -> None:
    """Python implementation of the minpack c-api hybrd test."""
    x, fvec, info, nfev = driver(trial_hybrd_fcn.address)
    _check_result(x, fvec, info, nfev)


def test_udata_hybrd1() -> None:
    """Python implementation of a modified minpack c-api hybrd1 test."""
    for i in (UDATA, UDATA.ctypes.data):
        x, fvec, info = hybrd1(trial_hybrd_fcn_udata.address, X0, TOL, i)
        _check_result(x, fvec, info)


def test_udata_hybrd1_() -> None:
    """Python implementation a modified minpack c-api hybrd1 test."""
    for i in (UDATA, UDATA.ctypes.data):
        x = X0.copy()
        fvec = empty(N)
        _, _, info = hybrd1_(trial_hybrd_fcn_udata.address, N, x, fvec, TOL, WA, LWA, i)
        _check_result(x, fvec, info)


def test_udata_hybrd() -> None:
    """Python implementation a modified minpack c-api hybrd test."""
    address = trial_hybrd_fcn_udata.address
    args = TOL, 2000, 1, 1, 0.0, DIAG, 2, 100.0, 0
    for i in (UDATA, UDATA.ctypes.data):
        x0 = X0.copy()
        fvec = empty(N)
        x, fvec, _, _, _, nfev, info = hybrd(address, x0, *args, i)
        _check_result(x, fvec, info, nfev)


def test_udata_hybrd_() -> None:
    """Python implementation a modified minpack c-api hybrd test."""
    for i in (UDATA, UDATA.ctypes.data):
        x, fvec, info, nfev = driver(trial_hybrd_fcn_udata.address, i)
        _check_result(x, fvec, info, nfev)
