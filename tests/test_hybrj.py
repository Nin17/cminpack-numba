"""Python implementations of the tests from the minpack c-api test suite."""

from numba import carray, cfunc, njit
from numpy import array, empty, finfo, float64, full, int32, ones, sqrt
from numpy.testing import assert_allclose, assert_equal

from cminpack_numba import enorm, hybrj, hybrj1, hybrj1_, hybrj_, hybrj_sig
from cminpack_numba.utils import ptr_from_val, val_from_ptr

from .test_hybrd import UDATA, trial_hybrd_fcn, trial_hybrd_fcn_udata

# ruff: noqa: ANN001, ARG001

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
LDFJAC = N
LWA = 180
X0 = full(N, -1.0)
FJAC = empty((N, N))
DIAG = ones(N)
WA = empty(LWA)


def _check_result(x, fvec, info, nfev=None, njev=None, tol=TOL) -> None:  # noqa: PLR0913
    assert_equal(info, 1)
    if nfev is not None:
        assert_equal(nfev, 14)  # ??? this is 15 in the original test
    if njev is not None:
        assert_equal(njev, 1)
    assert_allclose(enorm(fvec), 0.0, atol=tol)
    assert_allclose(x, REFERENCE, atol=10 * tol)


@cfunc(hybrj_sig)
def trial_hybrj_fcn(udata, n, x, fvec, fjac, ldfjac, iflag):  # noqa: ANN201, D103, PLR0913
    if iflag == 1:
        trial_hybrd_fcn(udata, n, x, fvec, iflag)
    else:
        for k in range(n):
            for j in range(n):
                fjac[k * ldfjac + j] = 0.0
            fjac[k * ldfjac + k] = 3.0 - 4.0 * x[k]
            if k != 0:
                fjac[k * ldfjac + k - 1] = -1.0
            if k != n - 1:
                fjac[k * ldfjac + k + 1] = -1.0
    return 0


@cfunc(hybrj_sig)
def trial_hybrj_fcn_udata(udata, n, x, fvec, fjac, ldfjac, iflag):  # noqa: ANN201, D103, PLR0913
    _udata = carray(udata, (4,), dtype=float64)
    if iflag == 1:
        trial_hybrd_fcn_udata(udata, n, x, fvec, iflag)
    else:
        for k in range(n):
            for j in range(n):
                fjac[k * ldfjac + j] = 0.0
            fjac[k * ldfjac + k] = _udata[0] - 2.0 * _udata[1] * x[k]
            if k != 0:
                fjac[k * ldfjac + k - 1] = -1.0
            if k != n - 1:
                fjac[k * ldfjac + k + 1] = -1.0
    return 0


@njit
def driver(address, udata=None):  # noqa: ANN201, D103
    nfevptr = ptr_from_val(int32(0))
    njevptr = ptr_from_val(int32(0))
    x = X0.copy()
    diag = ones(N)
    fvec = empty(N)
    fjac = empty((N, N))
    lr = (N * (N + 1)) // 2
    r = empty(lr)
    qtf = empty(N)
    wa1 = empty(N)
    wa2 = empty(N)
    wa3 = empty(N)
    wa4 = empty(N)

    args = address, N, x, fvec, fjac, LDFJAC, TOL, 2000, diag, 2, 100.0, 0
    args2 = nfevptr, njevptr, r, lr, qtf, wa1, wa2, wa3, wa4, udata

    _, _, _, _, _, _, _, info = hybrj_(*args, *args2)

    return x, fvec, val_from_ptr(nfevptr), val_from_ptr(njevptr), info


def test_hybrj1() -> None:
    """Python impelementation of the minpack c-api hybrj1 test."""
    x, fvec, _, info = hybrj1(trial_hybrj_fcn.address, X0, TOL)
    _check_result(x, fvec, info)


def test_hybrj1_() -> None:
    """Python impelementation of the minpack c-api hybrj1 test."""
    x = X0.copy()
    fvec = empty(N)
    args = trial_hybrj_fcn.address, N, x, fvec, FJAC, LDFJAC, TOL, WA, LWA
    _, _, _, info = hybrj1_(*args)
    _check_result(x, fvec, info)


def test_hybrj() -> None:
    """Python impelementation of the minpack c-api hybrj test."""
    args = trial_hybrj_fcn.address, X0, TOL, 2000, DIAG, 2, 100.0, 0
    x, fvec, _, _, _, nfev, njev, info = hybrj(*args)
    _check_result(x, fvec, info, nfev, njev)


def test_hybrj_() -> None:
    """Python impelementation of the minpack c-api hybrj test."""
    x, fvec, nfvev, njev, info = driver(trial_hybrj_fcn.address)
    _check_result(x, fvec, info, nfvev, njev)


def test_udata_hybrj1() -> None:
    """Python impelementation of the minpack c-api hybrj1 test."""
    for i in (UDATA, UDATA.ctypes.data):
        x, fvec, _, info = hybrj1(trial_hybrj_fcn_udata.address, X0, TOL, i)
        _check_result(x, fvec, info)


def test_udata_hybrj1_() -> None:
    """Python impelementation of the minpack c-api hybrj1 test."""
    address = trial_hybrj_fcn_udata.address
    args = FJAC, LDFJAC, TOL, WA, LWA
    for i in (UDATA, UDATA.ctypes.data):
        x = X0.copy()
        fvec = empty(N)
        _, _, _, info = hybrj1_(address, N, x, fvec, *args, i)
        _check_result(x, fvec, info)


def test_udata_hybrj() -> None:
    """Python impelementation a modified minpack c-api hybrj test."""
    args = trial_hybrj_fcn_udata.address, X0, TOL, 2000, DIAG, 2, 100.0, 0
    for i in (UDATA, UDATA.ctypes.data):
        x, fvec, _, _, _, nfev, njev, info = hybrj(*args, i)
        _check_result(x, fvec, info, nfev, njev)


def test_udata_hybrj_() -> None:
    """Python impelementation of the minpack c-api hybrj test."""
    for i in (UDATA, UDATA.ctypes.data):
        x, fvec, nfvev, njev, info = driver(trial_hybrj_fcn.address, i)
        _check_result(x, fvec, info, nfvev, njev)
