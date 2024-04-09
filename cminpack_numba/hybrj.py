"""Wrappers of the `hybrj` and `hybrj1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, floating, int32, ones

from ._cminpack import Cminpack
from .utils import _check_dtype, ptr_from_val, ptr_int32, val_from_ptr

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray

# -------------------------------------- hybrj1 -------------------------------------- #


def _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_hybrj1)
def _hybrj1_overload(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    hybrj1_external = Cminpack.hybrj1(x.dtype)

    @extending.register_jitable
    def impl(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata):
        info = hybrj1_external(
            fcn,
            udata,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            wa.ctypes,
            lwa,
        )
        return x, fvec, fjac, info

    if isinstance(udata, types.Array):
        return lambda fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata: impl(
            fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            tol,
            wa,
            lwa,
            udata.ctypes,
        )

    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata: impl(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        tol,
        wa,
        lwa,
        0,
    )


@njit
def hybrj1_(
    fcn: int64,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    fjac: NDArray[floating],
    ldfjac: int32,
    tol: floating,
    wa: NDArray[floating],
    lwa: int32,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], int]:
    # TODO(nin17): docstring.
    """.

    Parameters
    ----------
    fcn : int64
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
    wa : NDArray[floating]
        _description_
    lwa : int32
        _description_
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[NDArray[floating], NDArray[floating], NDArray[floating], int]
        _description_

    """
    return _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, wa, lwa, udata)


@njit
def hybrj1(
    fcn: int64,
    x: NDArray[floating],
    tol: floating | None = None,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], int]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    x : NDArray[floating]
        _description_
    tol : floating | None, optional
        _description_, by default None
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[NDArray[floating], NDArray[floating], NDArray[floating], int]
        _description_

    """
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32((n * (3 * n + 13)) // 2)
    fvec = empty(n, dtype=x.dtype)
    fjac = empty((n, n), dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    return _hybrj1(fcn, n, x.copy(), fvec, fjac, n, tol, wa, lwa, udata)


# --------------------------------------- hybrj -------------------------------------- #


def _hybrj(
    fcn,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    xtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    njev,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    raise NotImplementedError


@extending.overload(_hybrj)
def _hybrj_overload(
    fcn,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    xtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    njev,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    hybrd_external = Cminpack.hybrj(x.dtype)

    @extending.register_jitable
    def impl(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        info = hybrd_external(
            fcn,
            udata,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            xtol,
            maxfev,
            diag.ctypes,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            r.ctypes,
            lr,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
        )
        _nfev = val_from_ptr(nfev)
        _njev = val_from_ptr(njev)
        return x, fvec, fjac, r, qtf, _nfev, _njev, info

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            xtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            r,
            lr,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            udata: impl(
                fcn,
                n,
                x,
                fvec,
                fjac,
                ldfjac,
                xtol,
                maxfev,
                diag,
                mode,
                factor,
                nprint,
                nfev,
                njev,
                r,
                lr,
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
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata: impl(
            fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            xtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            njev,
            r,
            lr,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            0,
        )
    )


@njit
def hybrj_(
    fcn: int64,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    fjac: NDArray[floating],
    ldfjac: int32,
    xtol: floating,
    maxfev: int32,
    diag: NDArray[floating],
    mode: int32,
    factor: floating,
    nprint: int32,
    nfev: ptr_int32,
    njev: ptr_int32,
    r: NDArray[floating],
    lr: int32,
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
    NDArray[floating],
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
    xtol : floating
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
    r : NDArray[floating]
        _description_
    lr : int32
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
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           NDArray[floating], int, int, int]
        _description_

    """
    return _hybrj(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )


@njit
def hybrj(
    fcn: int64,
    x: NDArray[floating],
    xtol: floating | None = None,
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
    NDArray[floating],
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
    x : NDArray[floating]
        _description_
    xtol : floating | None, optional
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
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           NDArray[floating], int, int, int, ]
        _description_

    """
    n = int32(x.size)
    fvec = empty(n, dtype=x.dtype)
    nfevptr = ptr_from_val(int32(0))
    njevptr = ptr_from_val(int32(0))
    fjac = empty((n, n), dtype=x.dtype)
    ldfjac = n
    lr = (n * (n + 1)) // 2
    r = empty(lr, dtype=x.dtype)
    qtf = empty(n, dtype=x.dtype)
    wa = empty(4 * n, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n : 2 * n]
    wa3 = wa[2 * n : 3 * n]
    wa4 = wa[3 * n :]

    xtol = xtol or 1.49012e-8
    maxfev = maxfev or 200 * (n + 1)
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0

    return _hybrj(
        fcn,
        n,
        x.copy(),
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        nfevptr,
        njevptr,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
