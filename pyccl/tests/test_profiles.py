import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef(200, 'critical')


def smoke_assert_prof_real(profile):
    sizes = [(0, 0),
             (2, 0),
             (0, 2),
             (2, 3),
             (1, 3),
             (3, 1)]
    shapes = [(),
              (2,),
              (2,),
              (2, 3),
              (1, 3),
              (3, 1)]
    for (sr, sm), sh in zip(sizes, shapes):
        if sr == 0:
            r = 0.5
        else:
            r = np.linspace(0., 1., sr)
        if sm == 0:
            m = 1E12
        else:
            m = np.geomspace(1E10, 1E14, sm)
        p = profile._profile_real(COSMO, r, m, 1., M200)
        assert np.shape(p) == sh


def test_profile_defaults():
    p = ccl.halos.HaloProfile()
    with pytest.raises(NotImplementedError):
        p.profile_real(None, None, None, None)
    with pytest.raises(NotImplementedError):
        p.profile_fourier(None, None, None, None)


def test_profile_nfw_smoke():
    with pytest.raises(TypeError):
        p = ccl.halos.HaloProfileNFW(None)

    c = ccl.halos.ConcentrationDuffy08(M200)
    p = ccl.halos.HaloProfileNFW(c)
    smoke_assert_prof_real(p)
