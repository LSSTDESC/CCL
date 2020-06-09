import numpy as np
import pytest

import pyccl as ccl


@pytest.mark.parametrize('a', [
    1,
    1.,
    [1, 0.8],
    np.array([1, 0.8])])
@pytest.mark.parametrize('m', [
    [0.1, 0.8, 0.3],
    np.array([0.1, 0.8, 0.3])])
def test_omnuh2_smoke(a, m):
    om = ccl.Omeganuh2(a, m)
    assert np.all(np.isfinite(om))
    assert np.shape(om) == np.shape(a)


@pytest.mark.parametrize('split', ['normal', 'inverted', 'equal', 'sum',
                                   'single'])
def test_nu_masses_smoke(split):
    m = ccl.nu_masses(0.1, split)
    if split in ['sum', 'single']:
        assert np.ndim(m) == 0
    else:
        assert np.ndim(m) == 1
        assert np.shape(m) == (3,)


def test_neutrinos_raises():
    with pytest.raises(ValueError):
        ccl.nu_masses(0.1, 'blah')


@pytest.mark.parametrize('a', [
    1,
    1.])
@pytest.mark.parametrize('split', ['normal', 'inverted', 'equal'])
def test_nu_mass_consistency(a, split):
    m = ccl.nu_masses(0.1, split)
    assert np.allclose(ccl.Omeganuh2(a, m), 0.1, rtol=0, atol=1e-4)
