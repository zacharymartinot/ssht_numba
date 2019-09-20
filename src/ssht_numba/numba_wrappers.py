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
    "bad_meshgrid",
    "generate_dl",
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
    return delta / 2 - 1


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


@nb.njit
def bad_meshgrid(x, y):
    # output is the same as np.meshgrid(x,y)
    N = x.size
    M = y.size

    xx = np.empty((M, N), dtype=nb.float64)
    yy = np.empty((M, N), dtype=nb.float64)

    for i in range(M):
        xx[i, :] = x.copy()

    for j in range(N):
        yy[:, j] = y.copy()

    return xx, yy


@nb.njit
def dl_m(el, s, beta, delta):
    L = (delta.shape[2] + 1) / 2
    mp = np.arange(-el, el + 1)

    #     k = np.exp(1j*mp*beta)
    arg = mp * beta
    k = np.cos(arg) + 1j * np.sin(arg)

    ms = -el + L - 1
    mf = (el + 1) + (L - 1)
    s_i = -s + L - 1

    delta_1 = delta[el, ms:mf, ms:mf]
    delta_2 = delta[el, ms:mf, s_i]

    dl_m_out = np.zeros(2 * el + 1, dtype=nb.complex128)

    for i_m in range(len(mp)):
        dl_m_out[i_m] = 1j ** (-s - mp[i_m]) * np.sum(
            k * delta_1[:, i_m] * delta_2, axis=0
        )

    return dl_m_out


@nb.njit(parallel=True)
def ssht_numba_series_eval(f_lm, s, L, delta, theta, phi):
    f = np.zeros(len(theta), dtype=nb.complex128)

    spin_sign = (-1.0) ** s
    for i in nb.prange(len(theta)):
        for el in range(L):
            m_axis = np.arange(-el, el + 1)

            phases = m_axis * phi[i]
            sY_elm = (
                spin_sign
                * np.sqrt((2.0 * el + 1.0) / 4.0 / np.pi)
                * (np.cos(phases) + 1j * np.sin(phases))
            )
            sY_elm *= dl_m(el, s, theta[i], delta)

            j0 = el * (el + 1) - el
            j1 = el * (el + 1) + el

            f[i] += np.sum(sY_elm * f_lm[j0 : j1 + 1])

    return f

ssht_dl_beta_risbo_half_table = _ssht_cffi.lib.ssht_dl_beta_risbo_half_table


@nb.njit
def generate_dl(beta: float, L: int):

    dl_array = np.zeros((L, 2 * L - 1, 2 * L - 1))
    dl_dummy = np.zeros((2 * L - 1, 2 * L - 1))

    sqrt_tbl = np.sqrt(np.arange(0, 2 * (L - 1) + 1))
    signs = np.ones((L + 1, 1))

    offset_m = L - 1

    for i in range(1, L + 1, 2):
        signs[i] = -1

        dl_size = 2

        # do recursion
        # el = 0 first
        ssht_dl_beta_risbo_half_table(
            ffi.from_buffer(dl_dummy),
            beta,
            L,
            dl_size,
            0,
            ffi.from_buffer(sqrt_tbl),
            ffi.from_buffer(signs),
        )

        dl_array[0, offset_m, offset_m] = dl_dummy[offset_m, offset_m]

        for el in range(1, L):
            ssht_dl_beta_risbo_half_table(
                ffi.from_buffer(dl_dummy),
                beta,
                L,
                dl_size,
                el,
                ffi.from_buffer(sqrt_tbl),
                ffi.from_buffer(signs),
            )
            for i in range(-el, el + 1):
                for j in range(-el, el + 1):
                    dl_array[el, offset_m + i, offset_m + j] = dl_dummy[
                        offset_m + i, offset_m + j
                    ]

    return dl_array
