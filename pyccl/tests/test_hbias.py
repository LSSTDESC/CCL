import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
HBFS = [ccl.halos.HaloBiasSheth99,
        ccl.halos.HaloBiasSheth01,
        ccl.halos.HaloBiasTinker10,
        ccl.halos.HaloBiasBhattacharya11]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')
MVIR = ccl.halos.MassDef('vir', 'critical')
MDFS = [MVIR, MVIR, MFOF, MVIR]


@pytest.mark.parametrize('bM_class', HBFS)
def test_bM_subclasses_smoke(bM_class):
    bM = bM_class()
    for m in MS:
        b = bM(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


@pytest.mark.parametrize('bM_pair', zip(HBFS, MDFS))
def test_bM_mdef_raises(bM_pair):
    bM_class, mdef = bM_pair
    with pytest.raises(ValueError):
        bM_class(mass_def=mdef)


def test_bM_SO_allgood():
    bM = ccl.halos.HaloBiasTinker10(mass_def=MVIR)
    for m in MS:
        b = bM(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


@pytest.mark.parametrize('name', ['Tinker10', 'Sheth99'])
def test_bM_from_string(name):
    bM_class = ccl.halos.HaloBias.from_name(name)
    bM = bM_class()
    for m in MS:
        b = bM(COSMO, m, 0.9)
        assert np.all(np.isfinite(b))
        assert np.shape(b) == np.shape(m)


def test_bM_from_string_raises():
    with pytest.raises(KeyError):
        ccl.halos.HaloBias.from_name('Tinker11')
