"""A numba wrapper for the cminpack library."""

__version__ = "0.1.0"


# TODO(nin17): import functions for implementing bounds

from .chkder import chkder
from .dpmpar import dpmpar, sdpmpar
from .enorm import enorm, enorm_
from .hybrd import hybrd, hybrd1, hybrd1_, hybrd_
from .hybrj import hybrj, hybrj1, hybrj1_, hybrj_
from .lmder import lmder, lmder1, lmder1_, lmder_
from .lmdif import lmdif, lmdif1, lmdif1_, lmdif_
from .lmstr import lmstr, lmstr1, lmstr1_, lmstr_
from .signatures import (
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
from .utils import address_as_void_pointer, check_cfunc, ptr_from_val, val_from_ptr

__all__ = [
    # Utils
    "address_as_void_pointer",
    "check_cfunc",
    "ptr_from_val",
    "val_from_ptr",
    # Signatures
    "hybrd_sig",
    "shybrd_sig",
    "hybrj_sig",
    "shybrj_sig",
    "lmdif_sig",
    "slmdif_sig",
    "lmder_sig",
    "slmder_sig",
    "lmstr_sig",
    "slmstr_sig",
    "CminpackSignature",
    # Functions
    "enorm",
    "enorm_",
    "chkder",
    "dpmpar",
    "sdpmpar",
    "hybrd1",
    "hybrd1_",
    "hybrd",
    "hybrd_",
    "hybrj1",
    "hybrj1_",
    "hybrj",
    "hybrj_",
    "lmdif1",
    "lmdif1_",
    "lmdif",
    "lmdif_",
    "lmder1",
    "lmder1_",
    "lmder",
    "lmder_",
    "lmstr1",
    "lmstr1_",
    "lmstr",
    "lmstr_",
]
