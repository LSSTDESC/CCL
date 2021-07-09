"""
This is a catch-all unit test for the `pyccl.pyutils` module
which contains the `warn_api` and `deprecate_attr` decorators.
"""
import pyccl as ccl
from pyccl.errors import CCLWarning
import pytest

COSMO = ccl.CosmologyVanillaLCDM()
M200 = ccl.halos.MassDef200c()
CON = ccl.halos.ConcentrationDuffy08()
HMF = ccl.halos.MassFuncTinker10(COSMO)
HBF = ccl.halos.HaloBiasTinker10(COSMO)
HMC = ccl.halos.HMCalculator(COSMO, mass_function=HMF,
                             halo_bias=HBF, mass_def=M200)
PROF = ccl.halos.HaloProfileHOD(c_m_relation=CON)
COV = ccl.halos.Profile2pt()
COVh = ccl.halos.Profile2ptHOD()
PK2D = ccl.boltzmann.get_camb_pk_lin(COSMO)


def test_API_preserve_warnings():
    # 0. no warnings for the following exemplary functions
    with pytest.warns(None) as w_rec:
        R1 = ccl.Omega_nu_h2(1., m_nu=[1, 1, 1], T_CMB=2.7)
        R2 = ccl.nu_masses(Om_nu_h2=0.05, mass_split="normal")
        R3 = ccl.correlation_multipole(COSMO, 1., ell=100, dist=1., beta=1.5)

        R4 = CON.mass_def
        R5 = HMF.mass_def
        R6 = HBF.mass_def
        R7 = HMC.mass_function

        R8 = COV.fourier_2pt(COSMO, 1., 1e14, 1., PROF, mass_def=M200)
        R9 = COVh.fourier_2pt(COSMO, 1., 1e14, 1., PROF, mass_def=M200)

        S1 = ccl.halos.halomod_bias_1pt(COSMO, HMC, 1., 1., PROF,
                                        normprof=False)
        S2 = ccl.halos.halomod_power_spectrum(COSMO, HMC, 1., 1., PROF,
                                              prof2=PROF, prof_2pt=None,
                                              normprof=False)
    assert len(w_rec) == 0

    # 1. renamed function
    with pytest.warns(ccl.CCLWarning):
        r11 = ccl.Omeganuh2(1., m_nu=[1, 1, 1], T_CMB=2.7)
    assert r11 == R1

    # 2. star operator introduced
    with pytest.warns(CCLWarning):
        r12 = ccl.Omega_nu_h2(1., [1, 1, 1], T_CMB=2.7)
    assert r12 == R1

    # 3. renamed argument
    with pytest.warns(CCLWarning):
        r2 = ccl.nu_masses(OmNuh2=0.05, mass_split="normal")
    assert all(r2 == R2)

    # 4. swapped order
    with pytest.warns(CCLWarning):
        r31 = ccl.correlation_multipole(COSMO, 1., 1.5, 100, 1.)
    assert r31 == R3

    # 5. renamed argument + swapped order
    with pytest.warns(CCLWarning) as w_rec:
        r32 = ccl.correlation_multipole(COSMO, 1., 1.5, 100, s=1.)
    assert len(w_rec) == 2
    assert r32 == R3

    # 6. renamed class attribute
    with pytest.warns(CCLWarning):
        r4 = CON.mdef
    assert r4 == R4

    with pytest.warns(CCLWarning):
        r5 = HMF.mdef
    assert r5 == R5

    with pytest.warns(CCLWarning):
        r6 = HBF.mdef
    assert r6 == R6

    with pytest.warns(CCLWarning):
        r7 = HMC._massfunc
    assert r7 == R7

    # 7. old API patch for positional arguments
    with pytest.warns(CCLWarning):
        r8 = COV.fourier_2pt(PROF, COSMO, 1., 1e14, 1., mass_def=M200)
    assert r8 == R8

    with pytest.warns(CCLWarning):
        r9 = COVh.fourier_2pt(PROF, COSMO, 1., 1e14, 1., mass_def=M200)
    assert r9 == R9

    # 8. halo profile normalization
    with pytest.warns(CCLWarning):
        s1 = ccl.halos.halomod_bias_1pt(COSMO, HMC, 1., 1., PROF)
    assert s1 == S1

    with pytest.warns(None) as w_rec:
        s2 = ccl.halos.halomod_power_spectrum(COSMO, HMC, 1., 1.,
                                              PROF, None, PROF)
    assert len(w_rec) == 2
    assert s2 == S2


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileNFW,
                          ccl.halos.HaloProfileEinasto,
                          ccl.halos.HaloProfileHernquist,
                          ccl.halos.HaloProfileHOD])
def test_renamed_attribute(prof_class):
    prof = prof_class(c_m_relation=CON)

    with pytest.warns(None) as w_rec:
        prof.c_m_relation
    assert len(w_rec) == 0

    with pytest.warns(CCLWarning):
        prof.cM
