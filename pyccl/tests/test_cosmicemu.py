import numpy as np
import pyccl as ccl
import pytest

cemu_iv = ccl.CosmicemuMTIVPk('tot')
cemu_ii = ccl.CosmicemuMTIIPk('tot')


@pytest.mark.parametrize('cemu', [cemu_ii, cemu_iv])
def test_cosmicemu_smoke(cemu):
    cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum=cemu)
    k, pk = cemu.get_pk_at_a(cosmo, 1.0)
    pk2 = cosmo.nonlin_matter_power(k, 1.0)
    assert np.allclose(pk, pk2, atol=0, rtol=1E-6)


@pytest.mark.parametrize('cemu', [cemu_ii, cemu_iv])
def test_cosmicemu_As_raises(cemu):
    cosmo = ccl.Cosmology(Omega_c=0.25,
                          Omega_b=0.05,
                          h=0.67, n_s=0.96,
                          A_s=2E-9)
    # CosmicEmu needs sigma8
    with pytest.raises(ValueError):
        cemu.get_pk_at_a(cosmo, 1.0)


@pytest.mark.parametrize('cemu', [cemu_ii, cemu_iv])
def test_cosmicemu_outbound(cemu):
    cosmo = ccl.Cosmology(Omega_c=0.25,
                          Omega_b=0.05,
                          h=0.67, n_s=0.96,
                          sigma8=1.5)
    # sigma8 out of bounds
    with pytest.raises(ValueError):
        cemu.get_pk_at_a(cosmo, 1.0)
