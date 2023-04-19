import numpy as np
import pyccl as ccl
import pytest


@pytest.mark.parametrize('func', [
    'nfw_profile_3d', 'nfw_profile_2d',
    'einasto_profile_3d', 'hernquist_profile_3d'])
@pytest.mark.parametrize('r', [
    1,
    1.,
    np.array([1, 2, 3]),
    [1, 2, 3]])
def test_haloprofile_smoke(func, r):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    a = 1
    c = 5
    mass = 1e14
    odelta = 200
    # These are all deprecated
    with pytest.warns(ccl.CCLDeprecationWarning):
        prof = getattr(ccl, func)(cosmo, c, mass, odelta, a, r)
    assert np.all(np.isfinite(prof))
    assert np.shape(prof) == np.shape(r)


def test_IA_halo_model():
    from pyccl.pyutils import assert_warns
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                          sigma8=0.83, n_s=0.96)
    k_arr = np.geomspace(1E-3, 1e3, 256)  # For evaluating
    hmd_200m = ccl.halos.MassDef200m()
    cM = ccl.halos.ConcentrationDuffy08(hmd_200m)

    # lmax too low
    assert_warns(ccl.CCLWarning,
                 ccl.halos.SatelliteShearHOD,
                 cM, lmax=1)

    # lmax too high
    assert_warns(ccl.CCLWarning,
                 ccl.halos.SatelliteShearHOD,
                 cM, lmax=14)

    # lmax odd
    assert (ccl.halos.SatelliteShearHOD(cM, lmax=7)
            .lmax) % 2 == 0

    # Run with b!={0,2}
    assert (ccl.halos.SatelliteShearHOD(cM, b=-1.9, lmax=12)
            ._angular_fl).shape == (6, 1)

    # non-zero central fraction HOD parameters
    assert_warns(ccl.CCLWarning,
                 ccl.halos.SatelliteShearHOD,
                 cM, fc_0=0.5)

    # Testing FFTLog accuracy vs simps and spline method.
    s_g_HOD1 = ccl.halos.SatelliteShearHOD(cM)
    s_g_HOD2 = ccl.halos.SatelliteShearHOD(cM, integration_method='simps')
    s_g_HOD3 = ccl.halos.SatelliteShearHOD(cM, integration_method='spline')
    s_g1 = s_g_HOD1._usat_fourier(cosmo, k_arr, 1e13, 1., hmd_200m)
    s_g2 = s_g_HOD2._usat_fourier(cosmo, k_arr, 1e13, 1., hmd_200m)
    s_g3 = s_g_HOD3._usat_fourier(cosmo, k_arr, 1e13, 1., hmd_200m)
    assert np.all(np.abs((s_g1 - s_g2) / s_g2)) > 0.05
    assert np.all(np.abs((s_g3 - s_g2) / s_g3)) > 0.05

    # Wrong integration method
    with pytest.raises(ValueError):
        ccl.halos.SatelliteShearHOD(cM,
                                    integration_method="something_else")


def test_prefactor():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                          sigma8=0.83, n_s=0.96)

    hmd_200m = ccl.halos.MassDef200m()
    cM = ccl.halos.ConcentrationDuffy08(hmd_200m)
    nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200m)
    bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd_200m)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m)

    profile = ccl.halos.HaloProfileNFW(cM)  # a simple profile
    assert np.all(np.abs(profile._get_prefactor(
        cosmo, 1., hmc)-1.0 < 1e-10))

    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(cM)
    assert sat_gamma_HOD._get_prefactor(
        cosmo, 1., hmc) > 0
