"""Signatures for the functions passed to the cminpack functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import types

if TYPE_CHECKING:
    from numba.core.typing import Signature

__all__ = [
    "CminpackSignature",
    "hybrd_sig",
    "hybrj_sig",
    "lmder_sig",
    "lmdif_sig",
    "lmstr_sig",
    "shybrd_sig",
    "shybrj_sig",
    "slmder_sig",
    "slmdif_sig",
    "slmstr_sig",
]


class CminpackSignature:
    """Signatures for the functions passed to the cminpack functions."""

    @staticmethod
    def hybrd(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for `fcn` argument of [hybrd][cminpack_numba.hybrd] like functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.intc,  # iflag
        )

    @staticmethod
    def hybrj(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for `fcn` argument of [hybrj][cminpack_numba.hybrj] like functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.int32,  # iflag
        )

    @staticmethod
    def lmdif(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for `fcn` argument of [lmdif][cminpack_numba.lmdif] like functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.intc,  # iflag
        )

    @staticmethod
    def lmder(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for `fcn` argument of [lmder][cminpack_numba.lmder] like functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.intc,  # ldfjac
            types.intc,  # iflag
        )

    @staticmethod
    def lmstr(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for `fcn` argument of [lmstr][cminpack_numba.lmstr] like functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjrow
            types.intc,  # iflag
        )


# __cminpack_type_fcn_nn__
hybrd_sig = CminpackSignature.hybrd()
"""
Signature for `fcn` argument of [hybrd][cminpack_numba.hybrd] &
[hybrd1][cminpack_numba.hybrd1] with double precision arrays.

(udata: void*, n: int32, x: float64*, fvec: float64*, iflag: int32) -> int32
"""
# __cminpack_type_fcn_nn__
shybrd_sig = CminpackSignature.hybrd(dtype=types.float32)
"""
Signature for `fcn` argument of [hybrd][cminpack_numba.hybrd] &
[hybrd1][cminpack_numba.hybrd1] with single precision arrays.

(udata: void*, n: int32, x: float32*, fvec: float32*, iflag: int32) -> int32
"""

# __cminpack_type_fcnder_nn__
hybrj_sig = CminpackSignature.hybrj()
"""
Signature for `fcn` argument of [hybrj][cminpack_numba.hybrj] &
[hybrj1][cminpack_numba.hybrj1] with double precision arrays.

(udata: void*, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    ldfjac: int32, iflag:int32) -> int32
"""

# __cminpack_type_fcnder_nn__
shybrj_sig = CminpackSignature.hybrj(dtype=types.float32)
"""
Signature for `fcn` argument of [hybrj][cminpack_numba.hybrj] &
[hybrj1][cminpack_numba.hybrj1] with single precision arrays.

(udata: void*, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    ldfjac: int32, iflag:int32) -> int32
"""

# __cminpack_type_fcn_mn__
lmdif_sig = CminpackSignature.lmdif()
"""
Signature for `fcn` argument of [lmdif][cminpack_numba.lmdif] &
[lmdif1][cminpack_numba.lmdif1] with double precision arrays.

(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, iflag: int32)
    -> int32
"""

# __cminpack_type_fcn_mn_s__
slmdif_sig = CminpackSignature.lmdif(dtype=types.float32)
"""
Signature for `fcn` argument of [lmdif][cminpack_numba.lmdif] &
[lmdif1][cminpack_numba.lmdif1] with single precision arrays.

(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, iflag: int32)
    -> int32
"""

# __cminpack_type_fcnder_mn__
lmder_sig = CminpackSignature.lmder()
"""
Signature for `fcn` argument of [lmder][cminpack_numba.lmder] &
[lmder1][cminpack_numba.lmder1] with double precision arrays.

(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    ldfjac: int32, iflag: int32) -> int32
"""

# __cminpack_type_fcnder_mn__
slmder_sig = CminpackSignature.lmder(dtype=types.float32)
"""
Signature for `fcn` argument of [lmder][cminpack_numba.lmder] &
[lmder1][cminpack_numba.lmder1] with single precision arrays.

(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    ldfjac: int32, iflag: int32) -> int32
"""

# __cminpack_type_fcnderstr_mn__
lmstr_sig = CminpackSignature.lmstr()
"""
Signature for `fcn` argument of [lmstr][cminpack_numba.lmstr] &
[lmstr1][cminpack_numba.lmstr1] with double precision arrays.

(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    iflag: int32) -> int32
"""

# __cminpack_type_fcnderstr_mn__
slmstr_sig = CminpackSignature.lmstr(dtype=types.float32)
"""
Signature for `fcn` argument of [lmstr][cminpack_numba.lmstr] &
[lmstr1][cminpack_numba.lmstr1] with single precision arrays.

(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    iflag: int32) -> int32
"""
