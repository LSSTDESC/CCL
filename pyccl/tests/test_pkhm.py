import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200c()
PNFW = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200))
PEIN = ccl.halos.HaloProfileEinasto(ccl.halos.ConcentrationDuffy08(M200))
PKC = ccl.halos.ProfileCovar()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0


def test_profcovar_smoke():
    uk_NFW = PNFW.fourier(COSMO, KK, MM, AA,
                          mass_def=M200)
    uk_EIN = PEIN.fourier(COSMO, KK, MM, AA,
                          mass_def=M200)
    # Variance
    cv_NN = PKC.fourier_covar(PNFW, COSMO, KK, MM, AA,
                              mass_def=M200)
    assert np.all(np.fabs((cv_NN - uk_NFW**2)) < 1E-10)

    # Covariance
    cv_NE = PKC.fourier_covar(PNFW, COSMO, KK, MM, AA,
                              prof_2=PEIN, mass_def=M200)
    assert np.all(np.fabs((cv_NE - uk_NFW * uk_EIN)) < 1E-10)


def test_profcovar_errors():
    # Wrong first profile
    with pytest.raises(TypeError):
        PKC.fourier_covar(None, COSMO, KK, MM, AA,
                          prof_2=None, mass_def=M200)

    # Wrong second profile
    with pytest.raises(TypeError):
        PKC.fourier_covar(PNFW, COSMO, KK, MM, AA,
                          prof_2=M200, mass_def=M200)
