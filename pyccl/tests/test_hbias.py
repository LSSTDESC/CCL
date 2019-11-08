import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
PKA = ccl.Pk2D(lambda k, a: np.log(a/k), cosmo=COSMO)
HBFS = [ccl.halos.HaloBiasSheth99,
        ccl.halos.HaloBiasSheth01,
        ccl.halos.HaloBiasTinker10,
        ccl.halos.HaloBiasBhattacharya11]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')


@pytest.mark.parametrize('bM_class', HBFS)
def test_bM_subclasses_smoke(bM_class):
    bM = bM_class(COSMO)
    for m in MS:
        b = bM.get_halo_bias(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))


def test_bM_mdef_raises():
    bM_class = ccl.halos.HaloBiasTinker10
    with pytest.raises(ValueError):
        bM_class(COSMO, MFOF)
