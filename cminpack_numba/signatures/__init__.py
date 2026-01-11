"""Signatures for functions passed to cminpack functions."""

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

from cminpack_numba.src.signatures import (
    CminpackSignature,
    hybrd_sig,
    hybrj_sig,
    lmder_sig,
    lmdif_sig,
    lmstr_sig,
    shybrd_sig,
    shybrj_sig,
    slmder_sig,
    slmdif_sig,
    slmstr_sig,
)
