"""
This is a catch-all unit test for the `pyccl.pyutils` module
which contains the `warn_api` and `deprecate_attr` decorators.
"""
import pyccl as ccl
import pytest

COSMO = ccl.CosmologyVanillaLCDM()
HMF = ccl.halos.MassFuncTinker10(COSMO)

def test_API_preserve_warnings():
    # 0. no warnings for the following examplary functions
    with pytest.warns(None) as w_rec:
        ccl.Omega_nu_h2(1., m_nu=[1, 1, 1], T_CMB=2.7)
        ccl.correlation_multipole(COSMO, 1., ell=100, dist=1., beta=1.5)
        HMF.mass_def
    assert len(w_rec) == 0

    # 1. renamed function
    with pytest.warns(ccl.CCLWarning):
        ccl.Omeganuh2(1., m_nu=[1, 1, 1], T_CMB=2.7)

    # 2. star operator introduced
    with pytest.warns(FutureWarning):
        ccl.Omega_nu_h2(1., [1, 1, 1], T_CMB=2.7)

    # 3. renamed argument
    with pytest.warns(FutureWarning):
        ccl.nu_masses(OmNuh2=0.05, mass_split="normal")

    # 4. swapped order
    with pytest.warns(FutureWarning):
        ccl.correlation_multipole(COSMO, 1., 1.5, 100, 1.)

    # 5. renamed argument + swapped order
    with pytest.warns(FutureWarning) as w_rec:
        ccl.correlation_multipole(COSMO, 1., 1.5, 100, s=1.)
    assert len(w_rec) == 2

    # 6. renamed class attribute
    with pytest.warns(FutureWarning):
        HMF.mdef
