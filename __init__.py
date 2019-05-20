import numpy as np
import numba as nb

from .numba_wrappers import *

@nb.njit
def dl_m(el, s, beta, delta):
    L = (delta.shape[2] +1)/2
    mp = np.arange(-el,el+1)

#     k = np.exp(1j*mp*beta)
    arg = mp*beta
    k = np.cos(arg) + 1j*np.sin(arg)

    ms = -el + L-1
    mf = (el +1) + (L-1)
    s_i = -s + L-1

    delta_1 = delta[el,ms:mf,ms:mf]
    delta_2 = delta[el,ms:mf,s_i]

    dl_m_out = np.zeros(2*el+1, dtype=nb.complex128)

    for i_m in range(len(mp)):
        dl_m_out[i_m] = 1j**(-s - mp[i_m]) * np.sum(k * delta_1[:,i_m] * delta_2, axis=0)

#     for i_m in range(len(mp)):
#         for i_mp in range(len(mp)):
#             dl_m_out[i_m] += 1j**(-s - mp[i_m]) * k[i_mp] * delta_1[i_mp,i_m] * delta_2[i_mp]

    return dl_m_out

@nb.njit(parallel=True)
def ssht_numba_series_eval(f_lm, s, L, delta, theta, phi):
    f = np.zeros(len(theta), dtype=nb.complex128)

    spin_sign = (-1.)**s
    for i in nb.prange(len(theta)):
        for el in range(L):
            m_axis = np.arange(-el, el+1)

#             sY_elm = spin_sign * np.sqrt((2.*el +1.)/4./np.pi) * np.exp(-1j*m_axis*phi[i])
            phases = m_axis*phi[i]
            sY_elm = spin_sign * np.sqrt((2.*el +1.)/4./np.pi) * (np.cos(phases) + 1j*np.sin(phases))
            sY_elm *= dl_m(el, s, theta[i], delta)

            j0 = el*(el+1) - el
            j1 = el*(el+1) + el

            f[i] += np.sum(sY_elm * f_lm[j0:j1+1])

    return f
