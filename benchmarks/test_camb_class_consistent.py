import pytest
import numpy as np

import pyccl as ccl


@pytest.mark.parametrize('kwargs', [
    dict(m_nu=[0.2, 0.1, 0.5], w0=-0.8, wa=0.2),
    dict(m_nu=[0.2, 0.1, 0.0]),
    dict(m_nu=0.5),
    dict(m_nu=0.5, Neff=4.046),
    dict(w0=-0.8, wa=0.2),
    dict()])
def testcamb_class_consistent(kwargs):
    c_camb = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.95, A_s=2e-9,
        transfer_function='boltzmann_camb', **kwargs)

    c_class = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.95, A_s=2e-9,
        transfer_function='boltzmann_camb', **kwargs)

    assert np.allclose(
        ccl.sigma8(c_camb), ccl.sigma8(c_class), atol=0, rtol=2e-3)

    a = 0.845
    k = np.logspace(-4, 1, 100)
    pk_camb = ccl.linear_matter_power(c_camb, k, a)
    pk_class = ccl.linear_matter_power(c_class, k, a)

    assert np.allclose(pk_camb, pk_class, atol=0, rtol=2e-3)
