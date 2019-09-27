import cffi
import numba as nb
import numpy as np

from .wrappers import dl_beta_risbo_half_table

ffi = cffi.FFI()

__all__ = ["bad_meshgrid", "ssht_numba_series_eval", "generate_dl"]


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


@nb.njit
def generate_dl(beta: float, L: int):
    dl_array = np.zeros((L, 2 * L - 1, 2 * L - 1))
    dl_dummy = np.zeros((2 * L - 1, 2 * L - 1))

    sqrt_tbl = np.sqrt(np.arange(0, 2 * (L - 1) + 1))
    signs = np.ones((L + 1, 1))
    signs[1 : L + 1 : 2] = -1

    offset_m = L - 1

    # do recursion
    # el = 0 first
    dl_beta_risbo_half_table(dl_dummy, beta, L, 0, sqrt_tbl, signs)
    dl_array[0, offset_m, offset_m] = dl_dummy[offset_m, offset_m]

    for el in range(1, L):
        dl_beta_risbo_half_table(dl_dummy, beta, L, el, sqrt_tbl, signs)
        for i in range(-el, el + 1):
            for j in range(-el, el + 1):
                dl_array[el, offset_m + i, offset_m + j] = dl_dummy[
                    offset_m + i, offset_m + j
                ]

    return dl_array
