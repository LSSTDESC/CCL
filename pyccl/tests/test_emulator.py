import numpy as np
import pytest
import pyccl as ccl
from pyccl.emulator import Bounds
from pyccl.pyutils import CCLWarning
import warnings


def test_bounds_raises_warns():
    bounds = {"a": [0, 1], "b": [1, 0]}
    with pytest.raises(ValueError):
        Bounds(bounds)

    bounds = {"a": [0, 1], "b": [0, 1]}
    proposal = {"a": 0, "b": -1}
    B = Bounds(bounds)
    with pytest.raises(ValueError):
        B.check_bounds(proposal)

    proposal = {"a": 0, "b": 0.5, "c": 1}
    with pytest.warns(CCLWarning):
        B.check_bounds(proposal)


def test_emulator_load_raises():
    with pytest.raises(NotImplementedError):
        ccl.Emulator()


def test_emulator_from_name_raises():
    with pytest.raises(ValueError):
        ccl.PowerSpectrumEmulator.from_name("hello_world")


def test_arico21_raises():
    cosmo = ccl.CosmologyVanillaLCDM(baryons_power_spectrum="arico21",
                                     extra_parameters=None)
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            # ignore Tensorflow-related warnings
            warnings.simplefilter("ignore")
            cosmo.compute_nonlin_power()


def test_arico21_linear_nonlin_equiv():
    knl = np.geomspace(0.1, 5, 64)
    extras = {"arico21": {'M_c': 14, 'eta': -0.3, 'beta': -0.22,
                          'M1_z0_cen': 10.5, 'theta_out': 0.25,
                          'theta_inn': -0.86, 'M_inn': 13.4}
              }
    cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum="arico21",
                                     baryons_power_spectrum="arico21",
                                     extra_parameters=extras)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cosmo.compute_nonlin_power()
    pk0 = cosmo.get_nonlin_power().eval(knl, 1, cosmo)

    pk1 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "arico21")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # NL + bar
        pk1 = cosmo.baryon_correct("arico21", pk1).eval(knl, 1, cosmo)

    assert np.allclose(pk0, pk1, rtol=5e-3)
