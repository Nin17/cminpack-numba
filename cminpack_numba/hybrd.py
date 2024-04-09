"""Wrappers of the `hybrd` and `hybrd1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, finfo, floating, int32, ones

from ._cminpack import Cminpack
from .utils import _check_dtype, ptr_from_val, ptr_int32, val_from_ptr

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray

# -------------------------------------- hybrd1 -------------------------------------- #


def _hybrd1(
    fcn,
    n,
    x,
    fvec,
    tol,
    wa,
    lwa,
    udata=None,
):
    raise NotImplementedError


@extending.overload(_hybrd1)
def _hybrd1_overload(fcn, n, x, fvec, tol, wa, lwa, udata):
    _check_dtype((fvec, wa), x.dtype)
    hybrd1_external = Cminpack.hybrd1(x.dtype)

    @extending.register_jitable
    def impl(fcn, n, x, fvec, tol, wa, lwa, udata):
        info = hybrd1_external(
            fcn,
            udata,
            n,
            x.ctypes,
            fvec.ctypes,
            tol,
            wa.ctypes,
            lwa,
        )
        return x, fvec, info

    if isinstance(udata, types.Array):
        return lambda fcn, n, x, fvec, tol, wa, lwa, udata: impl(
            fcn,
            n,
            x,
            fvec,
            tol,
            wa,
            lwa,
            udata.ctypes,
        )

    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, tol, wa, lwa, udata: impl(
        fcn,
        n,
        x,
        fvec,
        tol,
        wa,
        lwa,
        0,
    )


@njit
def hybrd1_(
    fcn: int,
    n: int,
    x: NDArray[floating],
    fvec: NDArray[floating],
    tol: float,
    wa: NDArray[floating],
    lwa: int,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], int]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int
        _description_
    n : int
        _description_
    x : NDArray[floating]
        _description_
    fvec : NDArray[floating]
        _description_
    tol : float
        _description_
    wa : NDArray[floating]
        _description_
    lwa : int
        _description_
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[NDArray[floating], NDArray[floating], int]
        _description_

    """
    return _hybrd1(fcn, n, x, fvec, tol, wa, lwa, udata)


@njit
def hybrd1(
    fcn: int,
    x: NDArray[floating],
    tol: float | None = None,
    udata: NDArray | None = None,
) -> tuple[NDArray[floating], NDArray[floating], int]:
    # TODO(nin17): docstring
    """.

    Parameters
    ----------
    fcn : int
        _description_
    x : NDArray[floating]
        _description_
    tol : float | None, optional
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
    lwa = int32((n * (3 * n + 13)) // 2)
    fvec = empty(n, dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    return _hybrd1(fcn, n, x.copy(), fvec, tol, wa, lwa, udata)


# --------------------------------------- hybrd -------------------------------------- #


def _hybrd(
    fcn: int64,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    xtol: floating,
    maxfev: int32,
    ml: int32,
    mu: int32,
    epsfcn: floating,
    diag: NDArray[floating],
    mode: int32,
    factor: floating,
    nprint: int32,
    nfev: NDArray[int32],
    fjac: NDArray[floating],
    ldfjac: int32,
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
    int,
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    int,
]:
    raise NotImplementedError


@extending.overload(_hybrd)
def _hybrd_overload(
    fcn,
    n,
    x,
    fvec,
    xtol,
    maxfev,
    ml,
    mu,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    nfev,
    fjac,
    ldfjac,
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
    hybrd_external = Cminpack.hybrd(x.dtype)

    @extending.register_jitable
    def impl(
        fcn,
        n,
        x,
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
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
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag.ctypes,
            mode,
            factor,
            nprint,
            nfev,
            fjac.ctypes,
            ldfjac,
            r.ctypes,
            lr,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
        )
        _nfev = val_from_ptr(nfev)
        return x, fvec, fjac, r, qtf, _nfev, info

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            n,
            x,
            fvec,
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            fjac,
            ldfjac,
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
                xtol,
                maxfev,
                ml,
                mu,
                epsfcn,
                diag,
                mode,
                factor,
                nprint,
                nfev,
                fjac,
                ldfjac,
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
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
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
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            nfev,
            fjac,
            ldfjac,
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
def hybrd_(
    fcn: int64,
    n: int32,
    x: NDArray[floating],
    fvec: NDArray[floating],
    xtol: floating,
    maxfev: int32,
    ml: int32,
    mu: int32,
    epsfcn: floating,
    diag: NDArray[floating],
    mode: int32,
    factor: floating,
    nprint: int32,
    nfev: ptr_int32,
    fjac: NDArray[floating],
    ldfjac: int32,
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
]:
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
    xtol : floating
        _description_
    maxfev : int32
        _description_
    ml : int32
        _description_
    mu : int32
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
           NDArray[floating], int, int]
        _description_

    """
    return _hybrd(
        fcn,
        n,
        x,
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfev,
        fjac,
        ldfjac,
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
def hybrd(
    fcn: int64,
    x: NDArray[floating],
    xtol: floating | None = None,
    maxfev: int32 | None = None,
    ml: int32 | None = None,
    mu: int32 | None = None,
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
    NDArray[floating],
    NDArray[floating],
    int,
    int,
]:
    # TODO(nin17): docstring.
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
    ml : int32 | None, optional
        _description_, by default None
    mu : int32 | None, optional
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
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           NDArray[floating], int, int, ]
        _description_

    """
    n = int32(x.size)
    fvec = empty(n, dtype=x.dtype)
    nfevptr = ptr_from_val(int32(0))
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
    ml = ml or n
    mu = mu or n
    epsfcn = epsfcn or finfo(x.dtype).eps
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0

    return _hybrd(
        fcn,
        n,
        x.copy(),
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        nfevptr,
        fjac,
        ldfjac,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
