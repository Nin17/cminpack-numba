"""Wrappers of the `lmdif` and `lmdif1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, finfo, floating, int32, ones

from ._cminpack import Cminpack
from .utils import _check_dtype, ptr_from_val, ptr_int32, val_from_ptr

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray

# -------------------------------------- lmdif1 -------------------------------------- #


def _lmdif1(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_lmdif1)
def _lmdif1_overload(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
    _check_dtype((fvec, wa), x.dtype)
    lmdif1_external = Cminpack.lmdif1(x.dtype)

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata):
        info = lmdif1_external(
            fcn,
            udata,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            tol,
            iwa.ctypes,
            wa.ctypes,
            lwa,
        )
        return x, fvec, info

    if isinstance(udata, types.Array):
        return lambda fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata: impl(
            fcn,
            m,
            n,
            x,
            fvec,
            tol,
            iwa,
            wa,
            lwa,
            udata.ctypes,
        )

    if udata is not types.none:
        return impl
    return lambda fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata: impl(
        fcn,
        m,
        n,
        x,
        fvec,
        tol,
        iwa,
        wa,
        lwa,
        0,
    )


@njit
def lmdif1_(
    fcn: int64,
    m: int32,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    tol: floating,
    iwa: NDArray[int32],
    wa: NDArray[floating],
    lwa: int32,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], int]:
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
    tol : floating
        _description_
    iwa : NDArray[int32]
        _description_
    wa : NDArray[floating]
        _description_
    lwa : int32
        _description_
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[NDArray[floating], NDArray[floating], int]
        _description_

    """
    return _lmdif1(fcn, m, n, x, fvec, tol, iwa, wa, lwa, udata)


@njit
def lmdif1(
    fcn: int64,
    m: int32,
    x: NDArray[floating],
    tol: floating | None = None,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], int]:
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
    tuple[NDArray[floating], NDArray[floating], int]
        _description_

    """
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32(m * n + 5 * n + m)
    fvec = empty(m, dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    iwa = empty(n, dtype=int32)
    return _lmdif1(fcn, m, n, x.copy(), fvec, tol, iwa, wa, lwa, udata)


# --------------------------------------- lmdif -------------------------------------- #


def _lmdif(
    fcn,
    m,
    n,
    x,
    fvec,
    ftol,
    xtol,
    gtol,
    maxfev,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    fjac,
    ldfjac,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    raise NotImplementedError


@extending.overload(_lmdif)
def _lmdif_overload(
    fcn,
    m,
    n,
    x,
    fvec,
    ftol,
    xtol,
    gtol,
    maxfev,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    fjac,
    ldfjac,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    lmdif_external = Cminpack.lmdif(x.dtype)

    @extending.register_jitable
    def impl(
        fcn,
        m,
        n,
        x,
        fvec,
        ftol,
        xtol,
        gtol,
        maxfev,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        info = lmdif_external(
            fcn,
            udata,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            ftol,
            xtol,
            gtol,
            maxfev,
            epsfcn,
            diag.ctypes,
            mode,
            factor,
            nprint,
            nfev,
            fjac.ctypes,
            ldfjac,
            ipvt.ctypes,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
        )
        _nfev = val_from_ptr(nfev)
        return x, fvec, fjac, ipvt, qtf, _nfev, info

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            m,
            n,
            x,
            fvec,
            ftol,
            xtol,
            gtol,
            maxfev,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            fjac,
            ldfjac,
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
                ftol,
                xtol,
                gtol,
                maxfev,
                epsfcn,
                diag,
                mode,
                factor,
                nprint,
                nfev,
                fjac,
                ldfjac,
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
        ftol,
        xtol,
        gtol,
        maxfev,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
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
            ftol,
            xtol,
            gtol,
            maxfev,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            fjac,
            ldfjac,
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
def lmdif_(
    fcn: int64,
    m: int32,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    ftol: floating,
    xtol: floating,
    gtol: floating,
    maxfev: int32,
    epsfcn: floating,
    diag: NDArray[floating],
    mode: int32,
    factor: floating,
    nprint: int32,
    nfev: ptr_int32,
    fjac: NDArray[floating],
    ldfjac: int32,
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
    ftol : floating
        _description_
    xtol : floating
        _description_
    gtol : floating
        _description_
    maxfev : int32
        _description_
    epsfcn : floating
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
    fjac : NDArray[floating]
        _description_
    ldfjac : int32
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
           NDArray[floating], int, int]
        _description_

    """
    return _lmdif(
        fcn,
        m,
        n,
        x,
        fvec,
        ftol,
        xtol,
        gtol,
        maxfev,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )


@njit
def lmdif(
    fcn: int64,
    m: int32,
    x: NDArray[floating],
    ftol: floating | None = None,
    xtol: floating | None = None,
    gtol: floating | None = None,
    maxfev: int32 | None = None,
    epsfcn: floating | None = None,
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
    epsfcn : floating | None, optional
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
           NDArray[floating], int, int]
        _description_

    """
    n = int32(x.size)
    fvec = empty(m, dtype=x.dtype)
    fjac = empty((m, n), dtype=x.dtype)
    nfevptr = ptr_from_val(int32(0))
    ldfjac = m
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
    epsfcn = epsfcn or finfo(x.dtype).eps
    maxfev = maxfev or 200 * (n + 1)
    return _lmdif(
        fcn,
        m,
        n,
        x.copy(),
        fvec,
        ftol,
        xtol,
        gtol,
        maxfev,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfevptr,
        fjac,
        ldfjac,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
