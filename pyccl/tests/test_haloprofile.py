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
    from pyccl.pyutils import assert_warns
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    a = 1
    c = 5
    mass = 1e14
    odelta = 200
    # These are all deprecated
    prof = assert_warns(
        ccl.CCLWarning,
        getattr(ccl, func), cosmo, c, mass, odelta, a, r)
    assert np.all(np.isfinite(prof))
    assert np.shape(prof) == np.shape(r)
