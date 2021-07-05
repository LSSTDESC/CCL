"""
This is a catch-all unit test for the `pyccl.pyutils` module
which contains the `warn_api` and `deprecate_attr` decorators.
"""
import pyccl as ccl
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
    # 0. no warnings for the following examplary functions
    with pytest.warns(None) as w_rec:
        ccl.Omega_nu_h2(1., m_nu=[1, 1, 1], T_CMB=2.7)
        ccl.correlation_multipole(COSMO, 1., ell=100, dist=1., beta=1.5)
        CON.mass_def
        HMF.mass_def
        HBF.mass_def
        HMC.mass_function
        COV.fourier_2pt(COSMO, 1., 1e14, 1., PROF, mass_def=M200)
        COVh.fourier_2pt(COSMO, 1., 1e14, 1., PROF, mass_def=M200)
        PK2D.eval(COSMO, 1., 1.)
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
        CON.mdef

    with pytest.warns(FutureWarning):
        HMF.mdef

    with pytest.warns(FutureWarning):
        HBF.mdef

    with pytest.warns(FutureWarning):
        HMC._massfunc

    # 7. old API patch for positional arguments
    with pytest.warns(FutureWarning):
        COV.fourier_2pt(PROF, COSMO, 1., 1e14, 1., mass_def=M200)

    with pytest.warns(FutureWarning):
        COVh.fourier_2pt(PROF, COSMO, 1., 1e14, 1., mass_def=M200)

    with pytest.warns(FutureWarning):
        # used to be (k, a, cosmo); now it's (cosmo, k, a)
        PK2D.eval(1., 1., COSMO)


@pytest.parametrize('prof_class',
                    [ccl.halos.HaloProfileNFW,
                     ccl.halos.HaloProfileEinasto,
                     ccl.halos.HaloProfileHernquist,
                     ccl.halos.HaloProfileHOD])
def test_renamed_attribute(prof_class):
    prof = prof_class(c_m_relation=CON)

    with pytest.warns(None) as w_rec:
        prof.c_m_relation
    assert len(w_rec) == 0

    with pytest.warns(FutureWarning):
        prof.cM
