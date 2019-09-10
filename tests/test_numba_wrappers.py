"""
Tests the numba wrapper functions against the equivalent pyssht functions
using random input data.
"""

import numpy as np
import pyssht

import ssht_numba as sshtn

L = 180
s = -1


def test_everything():
    # Test indexing functions
    ind2elm_check = [pyssht.ind2elm(i) == sshtn.ind2elm(i) for i in range(L * L)]

    assert all(ind2elm_check), "ind2elm functions do not match"

    elm2ind_check = [
        pyssht.elm2ind(el, m) == sshtn.elm2ind(el, m)
        for el in range(L)
        for m in range(-el, el)
    ]

    assert all(elm2ind_check), "elm2ind functions do not match"

    assert pyssht.sample_shape(L, Method="MW") == sshtn.mw_sample_shape(L)
    assert pyssht.sample_shape(L, Method="MWSS") == sshtn.mwss_sample_shape(L)

    py_theta, py_phi = pyssht.sample_positions(L, Method="MW", Grid=False)
    nb_theta, nb_phi = sshtn.mw_sample_positions(L)
    assert np.allclose(py_theta, nb_theta)
    assert np.allclose(py_phi, nb_phi)

    py_theta, py_phi = pyssht.sample_positions(L, Method="MWSS", Grid=False)
    nb_theta, nb_phi = sshtn.mwss_sample_positions(L)

    assert np.allclose(py_theta, nb_theta)
    assert np.allclose(py_phi, nb_phi)

    # Generate random flms (of complex signal).
    np.random.seed(89834)
    flm = np.random.randn(L * L) + 1j * np.random.randn(L * L)

    # Zero harmonic coefficients with el<|spin|.
    ind_min = np.abs(s) ** 2
    flm[0:ind_min] = 0.0 + 1j * 0.0

    # MW inverse complex transform
    f_py_mw = pyssht.inverse(flm, L, Spin=s, Method="MW")

    f_nb_mw = np.empty(sshtn.mw_sample_shape(L), dtype=np.complex128)
    sshtn.mw_inverse_sov_sym(flm, L, s, f_nb_mw)

    assert np.allclose(f_py_mw, f_nb_mw)

    # MW forward complex transform, recovering input
    rec_flm_py_mw = pyssht.forward(f_py_mw, L, Spin=s, Method="MW")

    rec_flm_nb_mw = np.empty(L * L, dtype=np.complex128)
    sshtn.mw_forward_sov_conv_sym(f_nb_mw, L, s, rec_flm_nb_mw)

    assert np.allclose(rec_flm_py_mw, rec_flm_nb_mw)
    assert np.allclose(rec_flm_nb_mw, flm)

    # MW forward real transform

    f_re = np.random.randn(*sshtn.mw_sample_shape(L))

    flm_py_re_mw = pyssht.forward(f_re, L, Spin=0, Method="MW", Reality=True)

    flm_nb_re_mw = np.empty(L * L, dtype=np.complex128)
    sshtn.mw_forward_sov_conv_sym_real(f_re, L, flm_nb_re_mw)

    assert np.allclose(flm_py_re_mw, flm_nb_re_mw)

    # MW inverse real transform
    rec_f_re_py = pyssht.inverse(flm_py_re_mw, L, Spin=0, Method="MW", Reality=True)

    rec_f_re_nb = np.empty(sshtn.mw_sample_shape(L), dtype=np.float64)
    sshtn.mw_inverse_sov_sym_real(flm_nb_re_mw, L, rec_f_re_nb)

    assert np.allclose(rec_f_re_py, rec_f_re_nb)
    # Note that rec_f_re_{py,nb} != f_re since f_re is not band-limited at L

    # MWSS invserse complex transform
    f_py_mwss = pyssht.inverse(flm, L, Spin=s, Method="MWSS", Reality=False)

    f_nb_mwss = np.empty(sshtn.mwss_sample_shape(L), dtype=np.complex128)
    sshtn.mw_inverse_sov_sym_ss(flm, L, s, f_nb_mwss)

    assert np.allclose(f_py_mwss, f_nb_mwss)

    # MWSS forward complex transform
    rec_flm_py_mwss = pyssht.forward(f_py_mwss, L, Spin=s, Method="MWSS", Reality=False)

    rec_flm_nb_mwss = np.empty(L * L, dtype=np.complex128)
    sshtn.mw_forward_sov_conv_sym_ss(f_nb_mwss, L, s, rec_flm_nb_mwss)

    assert np.allclose(rec_flm_py_mwss, rec_flm_nb_mwss)
    assert np.allclose(rec_flm_nb_mwss, flm)

    # MWSS forward real transform

    f_re2 = np.random.randn(*sshtn.mwss_sample_shape(L))

    flm_py_re_mwss = pyssht.forward(f_re2, L, Spin=0, Method="MWSS", Reality=True)

    flm_nb_re_mwss = np.empty(L * L, dtype=np.complex128)
    sshtn.mw_forward_sov_conv_sym_ss_real(f_re2, L, flm_nb_re_mwss)

    assert np.allclose(flm_py_re_mwss, flm_nb_re_mwss)

    # MWSS inverse real transform

    rec_f_re_py_mwss = pyssht.inverse(
        flm_py_re_mwss, L, Spin=0, Method="MWSS", Reality=True
    )

    rec_f_re_nb_mwss = np.empty(sshtn.mwss_sample_shape(L), dtype=np.float64)
    sshtn.mw_inverse_sov_sym_ss_real(flm_nb_re_mwss, L, rec_f_re_nb_mwss)

    assert np.allclose(rec_f_re_py_mwss, rec_f_re_nb_mwss)
