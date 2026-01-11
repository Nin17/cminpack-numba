"""Utility functions for cminpack-numba."""

__all__ = [
    "address_as_void_pointer",
    "check_cfunc",
    "ptr_from_val",
    "val_from_ptr",
]

from cminpack_numba.src.utils import (
    address_as_void_pointer,
    check_cfunc,
    ptr_from_val,
    val_from_ptr,
)
