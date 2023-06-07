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
