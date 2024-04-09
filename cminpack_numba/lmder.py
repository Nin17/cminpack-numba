"""Wrappers of the `lmder` and `lmder1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, floating, int32, ones

from ._cminpack import Cminpack
from .utils import _check_dtype, ptr_from_val, ptr_int32, val_from_ptr

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray

# -------------------------------------- lmder1 -------------------------------------- #


def _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_lmder1)
def _lmder1_overload(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    tol,
    ipvt,
    wa,
    lwa,
    udata,
):
    _check_dtype((fvec, fjac, wa), x.dtype)
    lmder1_external = Cminpack.lmder1(x.dtype)

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata):
        info = lmder1_external(
            fcn,
            udata,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            ipvt.ctypes,
            wa.ctypes,
            lwa,
        )
        return x, fvec, fjac, ipvt, info

    if isinstance(udata, types.Array):
        return lambda fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata: impl(
            fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            tol,
            ipvt,
            wa,
            lwa,
            udata.ctypes,
        )
    if udata is not types.none:
        return impl
    return lambda fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata: impl(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        tol,
        ipvt,
        wa,
        lwa,
        0,
    )


@njit
def lmder1_(
    fcn: int64,
    m: int32,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    fjac: NDArray[floating],
    ldfjac: int32,
    tol: floating,
    ipvt: NDArray[int32],
    wa: NDArray[floating],
    lwa: int32,
    udata: NDArray | None = None,
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[int32],
    int,
]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    m : int32
        _description_
    n : int32
        _description_
    x : NDArray[floating]
        _description_
    fvec : NDArray[floating]
        _description_
    fjac : NDArray[floating]
        _description_
    ldfjac : int32
        _description_
    tol : floating
        _description_
    ipvt : NDArray[int32]
        _description_
    wa : NDArray[floating]
        _description_
    lwa : int32
        _description_
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[int32], int]
        _description_

    """
    return _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, ipvt, wa, lwa, udata)


@njit
def lmder1(
    fcn: int64,
    m: int32,
    x: NDArray[floating],
    tol: floating | None = None,
    udata: NDArray | None = None,
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[int32],
    int,
]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    m : int32
        _description_
    x : NDArray[floating]
        _description_
    tol : floating | None, optional
        _description_, by default None
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[int32], int]
        _description_

    """
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32(5 * n + m)
    fvec = empty(m, dtype=x.dtype)
    fjac = empty((m, n), dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    ipvt = empty(n, dtype=int32)
    return _lmder1(
        fcn,
        m,
        n,
        x.copy(),
        fvec,
        fjac,
        m,
        tol,
        ipvt,
        wa,
        lwa,
        udata,
    )


# --------------------------------------- lmder -------------------------------------- #


def _lmder(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    ftol,
    xtol,
    gtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    njev,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    raise NotImplementedError


@extending.overload(_lmder)
def _lmder_overload(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    ftol,
    xtol,
    gtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    njev,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    lmder_external = Cminpack.lmder(x.dtype)

    @extending.register_jitable
    def impl(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        info = lmder_external(
            fcn,
            udata,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag.ctypes,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            ipvt.ctypes,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
        )
        _nfev = val_from_ptr(nfev)
        _njev = val_from_ptr(njev)
        return x, fvec, fjac, ipvt, qtf, _nfev, _njev, info

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            ipvt,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            udata: impl(
                fcn,
                m,
                n,
                x,
                fvec,
                fjac,
                ldfjac,
                ftol,
                xtol,
                gtol,
                maxfev,
                diag,
                mode,
                factor,
                nprint,
                nfev,
                njev,
                ipvt,
                qtf,
                wa1,
                wa2,
                wa3,
                wa4,
                udata.ctypes,
            )
        )
    if udata is not types.none:
        return impl
    return (
        lambda fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata: impl(
            fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            ipvt,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            0,
        )
    )


@njit
def lmder_(
    fcn: int64,
    m: int32,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    fjac: NDArray[floating],
    ldfjac: int32,
    ftol: floating,
    xtol: floating,
    gtol: floating,
    maxfev: int32,
    diag: NDArray[floating],
    mode: int32,
    factor: floating,
    nprint: int32,
    nfev: ptr_int32,
    njev: ptr_int32,
    ipvt: NDArray[int32],
    qtf: NDArray[floating],
    wa1: NDArray[floating],
    wa2: NDArray[floating],
    wa3: NDArray[floating],
    wa4: NDArray[floating],
    udata: NDArray | None = None,
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[int32],
    NDArray[floating],
    int,
    int,
    int,
]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    m : int32
        _description_
    n : int32
        _description_
    x : NDArray[floating]
        _description_
    fvec : NDArray[floating]
        _description_
    fjac : NDArray[floating]
        _description_
    ldfjac : int32
        _description_
    ftol : floating
        _description_
    xtol : floating
        _description_
    gtol : floating
        _description_
    maxfev : int32
        _description_
    diag : NDArray[floating]
        _description_
    mode : int32
        _description_
    factor : floating
        _description_
    nprint : int32
        _description_
    nfev : ptr_int32
        _description_
    njev : ptr_int32
        _description_
    ipvt : NDArray[int32]
        _description_
    qtf : NDArray[floating]
        _description_
    wa1 : NDArray[floating]
        _description_
    wa2 : NDArray[floating]
        _description_
    wa3 : NDArray[floating]
        _description_
    wa4 : NDArray[floating]
        _description_
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[int32],
           NDArray[floating], int, int, int]
        _description_

    """
    return _lmder(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )


@njit
def lmder(
    fcn: int64,
    m: int32,
    x: NDArray[floating],
    ftol: floating | None = None,
    xtol: floating | None = None,
    gtol: floating | None = None,
    maxfev: int32 | None = None,
    diag: NDArray[floating] | None = None,
    mode: int32 | None = None,
    factor: floating | None = None,
    nprint: int32 | None = None,
    udata: NDArray | None = None,
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[int32],
    NDArray[floating],
    int,
    int,
    int,
]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    m : int32
        _description_
    x : NDArray[floating]
        _description_
    ftol : floating | None, optional
        _description_, by default None
    xtol : floating | None, optional
        _description_, by default None
    gtol : floating | None, optional
        _description_, by default None
    maxfev : int32 | None, optional
        _description_, by default None
    diag : NDArray[floating] | None, optional
        _description_, by default None
    mode : int32 | None, optional
        _description_, by default None
    factor : floating | None, optional
        _description_, by default None
    nprint : int32 | None, optional
        _description_, by default None
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[int32],
           NDArray[floating], int, int, int]
        _description_

    """
    n = int32(x.size)
    fvec = empty(m, dtype=x.dtype)
    fjac = empty((m, n), dtype=x.dtype)
    ldfjac = m
    nfevptr = ptr_from_val(int32(0))
    njevptr = ptr_from_val(int32(0))
    ipvt = empty(n, dtype=int32)
    qtf = empty(n, dtype=x.dtype)
    wa = empty(3 * n + m, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n : 2 * n]
    wa3 = wa[2 * n : 3 * n]
    wa4 = wa[3 * n :]

    ftol = ftol or 1.49012e-8
    xtol = xtol or 1.49012e-8
    gtol = gtol or 0.0
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    maxfev = maxfev or 200 * (n + 1)

    return _lmder(
        fcn,
        m,
        n,
        x.copy(),
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfevptr,
        njevptr,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
