import numpy as np
import pytest
from . import pyccl as ccl
from . import EmulatorObject
import warnings


def test_bounds_raises_warns():
    # malformed bounds
    bounds = {"a": [0, 1], "b": [1, 0]}
    with pytest.raises(ValueError):
        EmulatorObject(model=None, bounds=bounds)

    # out of bounds
    bounds = {"a": [0, 1], "b": [0, 1]}
    proposal = {"a": 0, "b": -1}
    emu = EmulatorObject(model=None, bounds=bounds)
    with pytest.raises(ValueError):
        emu.check_bounds(proposal)


def test_emulator_from_name_raises():
    # emulator does not exist
    with pytest.raises(ValueError):
        ccl.PowerSpectrumEmulator.from_name("hello_world")


def test_bacco_smoke():
    cosmo1 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96,
                           transfer_function="bacco")
    cosmo2 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           A_s=2.2315e-9, n_s=0.96,
                           transfer_function="bacco")
    assert np.allclose(cosmo1.sigma8(), cosmo2.sigma8(), rtol=1e-4)
    assert np.allclose(cosmo1.linear_matter_power(1, 1),
                       cosmo2.linear_matter_power(1, 1), rtol=2e-4)


def test_bacco_baryon_smoke():
    cosmo = ccl.CosmologyVanillaLCDM(baryons_power_spectrum="bacco",
                                     extra_parameters=None)
    with warnings.catch_warnings():
        # ignore Tensorflow-related warnings
        warnings.simplefilter("ignore")
        cosmo.compute_nonlin_power()


def test_bacco_linear_nonlin_equiv():
    # In this test we get the baryon-corrected NL power spectrum directly
    # from cosmo, and compare it with the NL where we have applied the
    # baryon correction afterwards.
    knl = np.geomspace(0.1, 5, 64)
    extras = {"bacco": {'M_c': 14, 'eta': -0.3, 'beta': -0.22,
                        'M1_z0_cen': 10.5, 'theta_out': 0.25,
                        'theta_inn': -0.86, 'M_inn': 13.4}
              }
    cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum="bacco",
                                     baryons_power_spectrum="bacco",
                                     extra_parameters=extras)
    with warnings.catch_warnings():
        # filter Pk2D narrower range warning
        warnings.simplefilter("ignore")
        cosmo.compute_nonlin_power()
    pk0 = cosmo.get_nonlin_power().eval(knl, 1, cosmo)

    emu = ccl.PowerSpectrumEmulator.from_name("bacco")()
    with warnings.catch_warnings():
        # filter Pk2D narrower range warning
        warnings.simplefilter("ignore")
        pk1 = emu.get_pk_nonlin(cosmo)
        # NL + bar
        pk1 = cosmo.baryon_correct("bacco", pk1).eval(knl, 1, cosmo)

    assert np.allclose(pk0, pk1, rtol=5e-3)


def test_power_spectum_emulator_raises():
    # does not have a `get_pk_linear` method
    cosmo = ccl.CosmologyVanillaLCDM()

    class DummyEmu(ccl.PowerSpectrumEmulator):
        name = "dummy"

        def __init__(self):
            super().__init__()

        def _load_emu(self):
            pass

    with pytest.raises(NotImplementedError):
        emu = ccl.PowerSpectrumEmulator.from_name("dummy")()
        emu.get_pk_linear(cosmo)


def test_power_spectrum_emulator_baryon_raises():
    cosmo = ccl.CosmologyVanillaLCDM()

    from . import PowerSpectrumBACCO
    emu = ccl.PowerSpectrumEmulator.from_name("bacco")()
    with warnings.catch_warnings():
        # filter Pk2D narrower range warning
        warnings.simplefilter("ignore")
        pk = emu.get_pk_nonlin(cosmo)
    func = PowerSpectrumBACCO._get_baryon_boost
    delattr(PowerSpectrumBACCO, "_get_baryon_boost")
    with pytest.raises(NotImplementedError):
        emu.include_baryons(cosmo, pk)

    # reset the emulator methods
    setattr(PowerSpectrumBACCO, "_get_baryon_boost", func)
