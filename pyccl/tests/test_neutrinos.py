import numpy as np
import pytest
import pyccl as ccl


@pytest.mark.parametrize('split', ['normal', 'inverted', 'equal', 'sum',
                                   'single'])
def test_nu_masses_smoke(split):
    m = ccl.nu_masses(Omega_nu_h2=0.1, mass_split=split)
    if split == "sum":
        assert np.shape(m) == ()
    elif split == "single":
        assert np.shape(m) == (1,)
    else:
        assert np.shape(m) == (3,)


def test_neutrinos_raises():
    with pytest.raises(ValueError):
        ccl.nu_masses(Omega_nu_h2=0.1, mass_split='blah')


@pytest.mark.parametrize('mass_split', ['normal', 'inverted', 'equal',
                                        'single', 'sum'])
@pytest.mark.parametrize('kwargs', [{'m_nu': -0.006},
                                    {'Omega_nu_h2': -0.001}])
def test_nu_masses_negative(mass_split, kwargs):
    with pytest.raises(ValueError, match='non-negative'):
        ccl.nu_masses(mass_split=mass_split, **kwargs)


@pytest.mark.parametrize('m_nu', [[-0.01, 0.02, 0.03],
                                  np.array([0.01, -0.02, 0.03])])
def test_nu_masses_negative_list(m_nu):
    with pytest.raises(ValueError, match='non-negative'):
        ccl.nu_masses(m_nu=m_nu, mass_split='list')


@pytest.mark.parametrize('mass_split', ['normal', 'inverted', 'equal',
                                        'single'])
def test_cosmology_negative_neutrino_mass(mass_split):
    with pytest.raises(ValueError, match='non-negative'):
        ccl.CosmologyVanillaLCDM(m_nu=-0.006, mass_split=mass_split)
    with pytest.raises(ValueError, match='non-negative'):
        ccl.CosmologyVanillaLCDM(m_nu=[-0.01, 0.02, 0.03])


@pytest.mark.parametrize('mass_split', ['normal', 'inverted', 'equal',
                                        'single', 'sum'])
def test_nu_masses_non_negative(mass_split):
    assert ccl.nu_masses(m_nu=0., mass_split=mass_split).size == 0
    masses = ccl.nu_masses(m_nu=0.15, mass_split=mass_split)
    assert np.all(masses >= 0)
    assert np.isclose(np.sum(masses), 0.15, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(
        ccl.nu_masses(m_nu=[0., 0.01, 0.02], mass_split=mass_split),
        [0., 0.01, 0.02])
