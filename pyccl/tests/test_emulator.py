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
