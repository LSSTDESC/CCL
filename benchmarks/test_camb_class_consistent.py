import pytest
import numpy as np
import warnings

import pyccl as ccl


def test_camb_class_consistent_smoke(kwargs=None, pkerr=1e-3):
    kwargs = kwargs or {}
    print('kwargs:', kwargs)

    c_camb = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.95, A_s=2e-9,
        transfer_function='boltzmann_camb', **kwargs)

    c_class = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.95, A_s=2e-9,
        transfer_function='boltzmann_class', **kwargs)

    with warnings.catch_warnings():
        # We do some tests here with massive neutrinos, which currently raises
        # a warning.
        # XXX: Do you really want to be raising a warning for this?
        #      This seems spurious to me.  (MJ)
        warnings.simplefilter("ignore")
        rel_sigma8 = np.abs(ccl.sigma8(c_camb) / ccl.sigma8(c_class) - 1)

    a = 0.845
    k = np.logspace(-3, 1, 100)
    pk_camb = ccl.linear_matter_power(c_camb, k, a)
    pk_class = ccl.linear_matter_power(c_class, k, a)
    rel_pk = np.max(np.abs(pk_camb / pk_class - 1))

    print('rel err pk:', rel_pk)
    print('rel err sigma8:', rel_sigma8)

    assert rel_sigma8 < 3e-3
    assert rel_pk < pkerr


@pytest.mark.parametrize('kwargs', [
    dict(m_nu=0.5),
    dict(m_nu=[0.2, 0.1, 0.0]),
    dict(m_nu=0.5, Neff=4.046)])
def test_camb_class_consistent_nu(kwargs):
    test_camb_class_consistent_smoke(kwargs=kwargs, pkerr=6e-3)


@pytest.mark.parametrize('kwargs', [
    dict(w0=-0.9, wa=0.0),
    dict(w0=-0.9, wa=-0.1)])
def test_camb_class_consistent_de(kwargs):
    test_camb_class_consistent_smoke(kwargs=kwargs, pkerr=1e-3)


@pytest.mark.parametrize('kwargs', [
    dict(m_nu=[0.2, 0.1, 0.5], w0=-0.9, wa=-0.1)])
def test_camb_class_consistent_de_nu(kwargs):
    test_camb_class_consistent_smoke(kwargs=kwargs, pkerr=8e-3)
