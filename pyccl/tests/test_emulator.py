import numpy as np
import pytest
from . import pyccl as ccl
from . import Bounds
from . import CCLWarning
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


def test_bacco_smoke():
    cosmo1 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           sigma8=0.81, n_s=0.96,
                           transfer_function="bacco")
    cosmo2 = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                           A_s=2.2315e-9, n_s=0.96,
                           transfer_function="bacco")
    assert np.allclose(cosmo1.sigma8(), cosmo2.sigma8(), rtol=1e-4)
    assert np.allclose(cosmo1.linear_matter_power(1, 1),
                       cosmo2.linear_matter_power(1, 1), rtol=1e-4)


def test_bacco_baryon_smoke():
    cosmo = ccl.CosmologyVanillaLCDM(baryons_power_spectrum="bacco",
                                     extra_parameters=None)
    with warnings.catch_warnings():
        # ignore Tensorflow-related warnings
        warnings.simplefilter("ignore")
        cosmo.compute_nonlin_power()


def test_bacco_linear_nonlin_equiv():
    knl = np.geomspace(0.1, 5, 64)
    extras = {"bacco": {'M_c': 14, 'eta': -0.3, 'beta': -0.22,
                        'M1_z0_cen': 10.5, 'theta_out': 0.25,
                        'theta_inn': -0.86, 'M_inn': 13.4}
              }
    cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum="bacco",
                                     baryons_power_spectrum="bacco",
                                     extra_parameters=extras)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cosmo.compute_nonlin_power()
    pk0 = cosmo.get_nonlin_power().eval(knl, 1, cosmo)

    pk1 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "bacco")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # NL + bar
        pk1 = cosmo.baryon_correct("bacco", pk1).eval(knl, 1, cosmo)

    assert np.allclose(pk0, pk1, rtol=5e-3)


def test_power_spectum_emulator_raises():
    cosmo = ccl.CosmologyVanillaLCDM()

    class DummyEmu(ccl.PowerSpectrumEmulator):
        name = "dummy"

        def __init__(self):
            super().__init__()

        def _load(self):
            pass

    with pytest.raises(NotImplementedError):
        ccl.PowerSpectrumEmulator.get_pk_linear(cosmo, "dummy")


def test_power_spectrum_emulator_raises():
    from pyccl.boltzmann import PowerSpectrumBACCO
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bacco")
    func = PowerSpectrumBACCO._get_pk_linear
    delattr(PowerSpectrumBACCO, "_get_pk_linear")
    with pytest.raises(NotImplementedError):
        cosmo.compute_linear_power()
    setattr(PowerSpectrumBACCO, "_get_pk_linear", func)


def test_power_spectrum_emulator_get_pk_equiv():
    cosmo = ccl.CosmologyVanillaLCDM()
    knl = np.geomspace(0.1, 5, 64)

    from pyccl.boltzmann import PowerSpectrumBACCO
    pk0 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "bacco")
    func1 = PowerSpectrumBACCO._get_pk_nonlin
    delattr(PowerSpectrumBACCO, "_get_pk_nonlin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk1 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "bacco")
    assert np.allclose(pk0.eval(knl, 1, cosmo),
                       pk1.eval(knl, 1, cosmo),
                       rtol=5e-3)

    func2 = PowerSpectrumBACCO._get_nonlin_boost
    delattr(PowerSpectrumBACCO, "_get_nonlin_boost")
    with pytest.raises(NotImplementedError):
        ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "bacco")

    # reset the emulator methods
    setattr(PowerSpectrumBACCO, "_get_pk_nonlin", func1)
    setattr(PowerSpectrumBACCO, "_get_nonlin_boost", func2)


def test_power_spectrum_emulator_baryon_raises():
    cosmo = ccl.CosmologyVanillaLCDM()

    from pyccl.boltzmann import PowerSpectrumBACCO
    pk = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "bacco")
    func = PowerSpectrumBACCO._get_baryon_boost
    delattr(PowerSpectrumBACCO, "_get_baryon_boost")
    with pytest.raises(NotImplementedError):
        ccl.PowerSpectrumEmulator.include_baryons(cosmo, "bacco", pk)

    # reset the emulator methods
    setattr(PowerSpectrumBACCO, "_get_baryon_boost", func)
