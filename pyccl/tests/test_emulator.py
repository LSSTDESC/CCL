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
    from pyccl.boltzmann import PowerSpectrumArico21
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="arico21")
    func = PowerSpectrumArico21._get_pk_linear
    delattr(PowerSpectrumArico21, "_get_pk_linear")
    with pytest.raises(NotImplementedError):
        cosmo.compute_linear_power()
    setattr(PowerSpectrumArico21, "_get_pk_linear", func)


def test_power_spectrum_emulator_get_pk_equiv():
    cosmo = ccl.CosmologyVanillaLCDM()
    knl = np.geomspace(0.1, 5, 64)

    from pyccl.boltzmann import PowerSpectrumArico21
    pk0 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "arico21")
    func1 = PowerSpectrumArico21._get_pk_nonlin
    delattr(PowerSpectrumArico21, "_get_pk_nonlin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk1 = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "arico21")
    assert np.allclose(pk0.eval(knl, 1, cosmo),
                       pk1.eval(knl, 1, cosmo),
                       rtol=5e-3)

    func2 = PowerSpectrumArico21._get_nonlin_boost
    delattr(PowerSpectrumArico21, "_get_nonlin_boost")
    with pytest.raises(NotImplementedError):
        ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "arico21")

    # reset the emulator methods
    setattr(PowerSpectrumArico21, "_get_pk_nonlin", func1)
    setattr(PowerSpectrumArico21, "_get_nonlin_boost", func2)


def test_power_spectrum_emulator_baryon_raises():
    cosmo = ccl.CosmologyVanillaLCDM()

    from pyccl.boltzmann import PowerSpectrumArico21
    pk = ccl.PowerSpectrumEmulator.get_pk_nonlin(cosmo, "arico21")
    func = PowerSpectrumArico21._get_baryon_boost
    delattr(PowerSpectrumArico21, "_get_baryon_boost")
    with pytest.raises(NotImplementedError):
        ccl.PowerSpectrumEmulator.include_baryons(cosmo, "arico21", pk)

    # reset the emulator methods
    setattr(PowerSpectrumArico21, "_get_baryon_boost", func)


def test_power_spectrum_emulator_apply_model_equiv():
    cosmo = ccl.CosmologyVanillaLCDM()
    cosmo.compute_linear_power()
    pkl = cosmo.get_linear_power()
    knl = np.geomspace(0.1, 5, 64)

    from pyccl.boltzmann import PowerSpectrumArico21
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk0 = ccl.Pk2D.apply_model(cosmo, "arico21", pk_linear=pkl)
    func1 = PowerSpectrumArico21._get_nonlin_boost
    delattr(PowerSpectrumArico21, "_get_nonlin_boost")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk1 = ccl.Pk2D.apply_model(cosmo, "arico21", pk_linear=pkl)

    assert np.allclose(pk0.eval(knl, 1, cosmo),
                       pk1.eval(knl, 1, cosmo),
                       rtol=5e-3)

    func2 = PowerSpectrumArico21._get_pk_nonlin
    delattr(PowerSpectrumArico21, "_get_pk_nonlin")
    with pytest.raises(NotImplementedError):
        ccl.Pk2D.apply_model(cosmo, "arico21", pk_linear=pkl)

    # reset the emulator methods
    setattr(PowerSpectrumArico21, "_get_nonlin_boost", func1)
    setattr(PowerSpectrumArico21, "_get_pk_nonlin", func2)
