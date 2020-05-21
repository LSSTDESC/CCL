import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MF_EQUIV = {'tinker10': 'Tinker10',
            'tinker': 'Tinker08',
            'watson': 'Watson13',
            'shethtormen': 'Sheth99',
            'angulo': 'Angulo12'}
MF_TYPES = sorted(list(MF_EQUIV.keys()))


@pytest.mark.parametrize('mf_type', MF_TYPES)
def test_massfunc_models_smoke(mf_type):
    from pyccl.pyutils import assert_warns
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function=mf_type)
    hmf_cls = ccl.halos.mass_function_from_name(MF_EQUIV[mf_type])
    hmf = hmf_cls(cosmo)
    for m in MS:
        # Deprecated
        nm_old = assert_warns(ccl.CCLWarning, ccl.massfunc, cosmo, m, 1.)
        nm_new = hmf.get_mass_function(cosmo, m, 1.)
        assert np.all(np.isfinite(nm_old))
        assert np.shape(nm_old) == np.shape(m)
        assert np.all(np.array(nm_old) ==
                      np.array(nm_new))


@pytest.mark.parametrize('mf_type', ['tinker10', 'shethtormen'])
def test_halo_bias_models_smoke(mf_type):
    from pyccl.pyutils import assert_warns
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function=mf_type)
    hbf_cls = ccl.halos.halo_bias_from_name(MF_EQUIV[mf_type])
    hbf = hbf_cls(cosmo)
    for m in MS:
        # Deprecated
        bm_old = assert_warns(ccl.CCLWarning, ccl.halo_bias, cosmo, m, 1.)
        bm_new = hbf.get_halo_bias(cosmo, m, 1.)
        assert np.all(np.isfinite(bm_old))
        assert np.shape(bm_old) == np.shape(m)
        assert np.all(np.array(bm_old) ==
                      np.array(bm_new))


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_massfunc_smoke(m):
    a = 0.8
    mf = ccl.halos.MassFuncTinker10(COSMO).get_mass_function(COSMO, m, a)
    assert np.all(np.isfinite(mf))
    assert np.shape(mf) == np.shape(m)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_massfunc_m2r_smoke(m):
    from pyccl.pyutils import assert_warns
    # Deprecated
    # TODO: switch to mass2radius_lagrangian
    r = assert_warns(ccl.CCLWarning, ccl.massfunc_m2r, COSMO, m)
    assert np.all(np.isfinite(r))
    assert np.shape(r) == np.shape(m)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_sigmaM_smoke(m):
    a = 0.8
    s = ccl.sigmaM(COSMO, m, a)
    assert np.all(np.isfinite(s))
    assert np.shape(s) == np.shape(m)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_halo_bias_smoke(m):
    from pyccl.pyutils import assert_warns
    a = 0.8
    # Deprecated
    # TODO: switch to HaloBias
    b = assert_warns(ccl.CCLWarning, ccl.halo_bias, COSMO, m, a)
    assert np.all(np.isfinite(b))
    assert np.shape(b) == np.shape(m)
