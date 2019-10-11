"""Wrappers to the functions in the extension module built by `build_ssht_cffi.py`

The function signatures are defined in the SSHT docs.
"""
import cffi
import numba as nb
import numpy as np
from numba import cffi_support

from . import _ssht_cffi

ffi = cffi.FFI()

# allows numba to use functions from this module in nopython mode
cffi_support.register_module(_ssht_cffi)

# tells numba how to map numba types to the types defined in the cffi function signatures
cffi_support.register_type(ffi.typeof("double _Complex"), nb.types.complex128)
cffi_support.register_type(
    ffi.typeof("double _Complex *"), nb.types.CPointer(nb.types.complex128)
)

__all__ = [
    "mw_forward_sov_conv_sym",
    "mw_inverse_sov_sym",
    "mw_forward_sov_conv_sym_real",
    "mw_inverse_sov_sym_real",
    "mw_forward_sov_conv_sym_ss",
    "mw_inverse_sov_sym_ss",
    "mw_forward_sov_conv_sym_ss_real",
    "mw_inverse_sov_sym_ss_real",
    "ind2elm",
    "elm2ind",
    "mw_sample_shape",
    "mwss_sample_shape",
    "mw_sample_positions",
    "mwss_sample_positions",
    "dl_beta_risbo_half_table",
]

# assign references outside of the njit function, apparently can't use
# `_ssht_cffi.lib` attribute inside njit function? not sure why, but this makes it work
ssht_core_mw_forward_sov_conv_sym = _ssht_cffi.lib.ssht_core_mw_forward_sov_conv_sym


@nb.njit
def mw_forward_sov_conv_sym(f, L, s, flm):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_forward_sov_conv_sym(ptr_flm, ptr_f, L, s, 0, 0)


ssht_core_mw_inverse_sov_sym = _ssht_cffi.lib.ssht_core_mw_inverse_sov_sym


@nb.njit
def mw_inverse_sov_sym(flm, L, s, f):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_inverse_sov_sym(ptr_f, ptr_flm, L, s, 0, 0)


ssht_core_mw_forward_sov_conv_sym_real = (
    _ssht_cffi.lib.ssht_core_mw_forward_sov_conv_sym_real
)


@nb.njit
def mw_forward_sov_conv_sym_real(f, L, flm):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_forward_sov_conv_sym_real(ptr_flm, ptr_f, L, 0, 0)


ssht_core_mw_inverse_sov_sym_real = _ssht_cffi.lib.ssht_core_mw_inverse_sov_sym_real


@nb.njit
def mw_inverse_sov_sym_real(flm, L, f):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_inverse_sov_sym_real(ptr_f, ptr_flm, L, 0, 0)


ssht_core_mw_forward_sov_conv_sym_ss = (
    _ssht_cffi.lib.ssht_core_mw_forward_sov_conv_sym_ss
)


@nb.njit
def mw_forward_sov_conv_sym_ss(f, L, s, flm):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_forward_sov_conv_sym_ss(ptr_flm, ptr_f, L, s, 0, 0)


ssht_core_mw_inverse_sov_sym_ss = _ssht_cffi.lib.ssht_core_mw_inverse_sov_sym_ss


@nb.njit
def mw_inverse_sov_sym_ss(flm, L, s, f):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_inverse_sov_sym_ss(ptr_f, ptr_flm, L, s, 0, 0)


ssht_core_mw_forward_sov_conv_sym_ss_real = (
    _ssht_cffi.lib.ssht_core_mw_forward_sov_conv_sym_ss_real
)

@nb.njit
def mw_forward_sov_conv_sym_ss_real(f, L, flm):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_forward_sov_conv_sym_ss_real(ptr_flm, ptr_f, L, 0, 0)


ssht_core_mw_inverse_sov_sym_ss_real = (
    _ssht_cffi.lib.ssht_core_mw_inverse_sov_sym_ss_real
)

@nb.njit
def mw_inverse_sov_sym_ss_real(flm, L, f):
    ptr_flm = ffi.from_buffer(flm)
    ptr_f = ffi.from_buffer(f)

    ssht_core_mw_inverse_sov_sym_ss_real(ptr_f, ptr_flm, L, 0, 0)


@nb.njit
def elm2ind(el, m):
    return el * el + el + m


@nb.njit
def isqrt(n):
    square = 1
    delta = 3
    while square <= n:
        square += delta
        delta += 2
    return delta // 2 - 1


@nb.njit
def ind2elm(ind):
    el = isqrt(ind)
    m = ind - el * el - el
    return el, m


@nb.njit
def mw_sample_shape(L):
    return L, 2 * L - 1


@nb.njit
def mwss_sample_shape(L):
    return L + 1, 2 * L


@nb.njit
def gl_sample_shape(L):
    return L, 2 * L - 1


ssht_sampling_mw_t2theta = _ssht_cffi.lib.ssht_sampling_mw_t2theta
ssht_sampling_mw_p2phi = _ssht_cffi.lib.ssht_sampling_mw_p2phi


@nb.njit
def mw_sample_positions(L):
    n_theta, n_phi = mw_sample_shape(L)
    theta = np.zeros(n_theta, dtype=nb.float64)
    phi = np.zeros(n_phi, dtype=nb.float64)

    for t in range(n_theta):
        theta[t] = ssht_sampling_mw_t2theta(t, L)

    for p in range(n_phi):
        phi[p] = ssht_sampling_mw_p2phi(p, L)

    return theta, phi

ssht_sampling_mw_ss_t2theta = _ssht_cffi.lib.ssht_sampling_mw_ss_t2theta
ssht_sampling_mw_ss_p2phi = _ssht_cffi.lib.ssht_sampling_mw_ss_p2phi


@nb.njit
def mwss_sample_positions(L):
    n_theta, n_phi = mwss_sample_shape(L)
    theta = np.zeros(n_theta, dtype=nb.float64)
    phi = np.zeros(n_phi, dtype=nb.float64)

    for t in range(n_theta):
        theta[t] = ssht_sampling_mw_ss_t2theta(t, L)

    for p in range(n_phi):
        phi[p] = ssht_sampling_mw_ss_p2phi(p, L)

    return theta, phi


_ssht_dl_beta_risbo_half_table = _ssht_cffi.lib.ssht_dl_beta_risbo_half_table


@nb.njit
def dl_beta_risbo_half_table(
    dl_array: np.ndarray,
    beta: float,
    L: int,
    el: int,
    sqrt_tbl: np.ndarray,
    signs: np.ndarray,
):
    dl_size = 2
    _ssht_dl_beta_risbo_half_table(
        ffi.from_buffer(dl_array),
        beta,
        L,
        dl_size,
        el,
        ffi.from_buffer(sqrt_tbl),
        ffi.from_buffer(signs),
    )
