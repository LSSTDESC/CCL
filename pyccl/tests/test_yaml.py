import tempfile
import pytest
import filecmp
import io
import sys
from types import SimpleNamespace

import numpy as np
import yaml

import pyccl as ccl
from pyccl.cosmology import _make_yaml_friendly


def test_yaml():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.97, m_nu=[0.01, 0.2, 0.3],
                          transfer_function="boltzmann_camb",
                          )

    # Make temporary files
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile1, \
            tempfile.NamedTemporaryFile(delete=True) as tmpfile2:
        cosmo.write_yaml(tmpfile1.name)

        cosmo2 = ccl.Cosmology.read_yaml(tmpfile1.name)
        cosmo2.write_yaml(tmpfile2.name)

        # Compare the contents of the two files
        assert filecmp.cmp(tmpfile1.name, tmpfile2.name, shallow=False)
        # Compare the two Cosmology objects
        assert cosmo == cosmo2

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.97, m_nu=0.1, mass_split="equal",
                          transfer_function="boltzmann_camb")

    stream = io.StringIO()
    cosmo.write_yaml(stream)
    stream.seek(0)

    cosmo2 = ccl.Cosmology.read_yaml(stream)
    stream2 = io.StringIO()
    cosmo2.write_yaml(stream2)

    assert stream.getvalue() == stream2.getvalue()


@pytest.mark.parametrize('model', [
    ccl.baryons.BaryonsSchneider15(log10Mc=13.5, eta_b=0.7, k_s=42.),
    ccl.baryons.BaryonsvanDaalen19(fbar=0.6, mass_def='200c'),
])
def test_yaml_baryons_and_modified_gravity(model):
    cosmo = ccl.CosmologyVanillaLCDM(
        baryonic_effects=model,
        transfer_function='boltzmann_isitgr',
        mg_parametrization=ccl.modified_gravity.MuSigmaMG(
            mu_0=0.2, sigma_0=0.3, c1_mg=1.2, c2_mg=1.3, lambda_mg=0.4)
    )
    stream = io.StringIO()
    cosmo.write_yaml(stream)
    stream.seek(0)
    restored = ccl.Cosmology.read_yaml(stream)
    assert restored == cosmo
    assert restored.baryons is not model
    np.testing.assert_allclose(
        restored.baryons.boost_factor(restored, [0.1, 1.], 1.),
        model.boost_factor(cosmo, [0.1, 1.], 1.))


@pytest.mark.parametrize('emulator_class', [
    ccl.emulators.CosmicemuMTIIPk, ccl.emulators.CosmicemuMTIVPk])
@pytest.mark.parametrize('kind', ['tot', 'cb'])
def test_yaml_emulator(emulator_class, kind):
    emulator = emulator_class(kind=kind)
    cosmo = ccl.CosmologyVanillaLCDM(matter_power_spectrum=emulator)
    stream = io.StringIO()
    cosmo.write_yaml(stream)
    assert yaml.safe_load(stream.getvalue())["matter_power_spectrum"] == {
        "class": emulator_class.__name__, "parameters": {"kind": kind}}
    stream.seek(0)
    restored = ccl.Cosmology.read_yaml(stream)
    assert type(restored.nl_pk_emu) is emulator_class
    assert restored.nl_pk_emu.kind == kind
    assert restored.nl_pk_emu is not emulator
    k, pk = emulator.get_pk_at_a(cosmo, 1.)
    restored_k, restored_pk = restored.nl_pk_emu.get_pk_at_a(restored, 1.)
    np.testing.assert_array_equal(restored_k, k)
    np.testing.assert_allclose(restored_pk, pk)


def test_yaml_baccoemu_configuration(monkeypatch):
    # Exercise the CCL wrappers without downloading external emulator weights.
    calls = []

    def matter_powerspectrum(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(emulator={
            name: {'bounds': [[0.2, 1.]], 'k': [0.01, 5.]}
            for name in ['linear', 'nonlinear', 'baryon']})

    monkeypatch.setitem(sys.modules, 'baccoemu', SimpleNamespace(
        Matter_powerspectrum=matter_powerspectrum))
    linear = ccl.emulators.BaccoemuLinear()
    nonlinear = ccl.emulators.BaccoemuNonlinear(
        nonlinear_emu_path='/custom/emulator',
        nonlinear_emu_details='details.json', n_sampling_a=37)
    baryons = ccl.baryons.BaryonsBaccoemu(verbose=True)
    baryons.update_parameters(log10_M_c=13.7, log10_eta=-0.5)
    cosmo = ccl.CosmologyVanillaLCDM(
        transfer_function=linear, matter_power_spectrum=nonlinear,
        baryonic_effects=baryons)
    stream = io.StringIO()
    cosmo.write_yaml(stream)
    stream.seek(0)
    restored = ccl.Cosmology.read_yaml(stream)
    assert calls[:3] == calls[3:]
    assert isinstance(restored.lin_pk_emu, ccl.emulators.BaccoemuLinear)
    assert restored.nl_pk_emu.nonlinear_emu_path == '/custom/emulator'
    assert restored.nl_pk_emu.nonlinear_emu_details == 'details.json'
    assert restored.nl_pk_emu.n_sampling_a == 37
    assert restored.baryons == baryons
    assert restored.baryons.verbose is True


def test_yaml_unsupported_model():
    class CustomBaryons(ccl.baryons.BaryonsSchneider15):
        pass

    cosmo = ccl.CosmologyVanillaLCDM(baryonic_effects=CustomBaryons())
    with pytest.raises(ValueError, match='CustomBaryons'):
        cosmo.write_yaml(io.StringIO())


@pytest.mark.parametrize('model', [
    {'class': 'UnknownModel', 'parameters': {}},
    {'class': 'MuSigmaMG'},
    {'class': 'MuSigmaMG', 'parameters': []},
])
def test_yaml_invalid_model_and_override(model):
    cosmo = ccl.CosmologyVanillaLCDM()
    params = cosmo.to_dict()
    params['mg_parametrization'] = model
    text = yaml.safe_dump(params)
    with pytest.raises(ValueError, match='YAML model'):
        ccl.Cosmology.read_yaml(io.StringIO(text))
    # Apply overrides before trying to reconstruct an unavailable model.
    restored = ccl.Cosmology.read_yaml(
        io.StringIO(text), mg_parametrization=None)
    assert restored == cosmo


def test_to_dict():
    cosmo = ccl.CosmologyVanillaLCDM(
        transfer_function=ccl.emulators.EmulatorPk(),
        matter_power_spectrum=ccl.emulators.CosmicemuMTIIPk(),
        baryonic_effects=ccl.baryons.BaryonsvanDaalen19(),
        mg_parametrization=ccl.modified_gravity.MuSigmaMG()
    )

    assert cosmo == ccl.Cosmology(**cosmo.to_dict())

    # Check that all arguments to Cosmology are stored
    init_params = {k: v for k, v in cosmo.__signature__.parameters.items()
                   if k != "self"}
    assert set(cosmo.to_dict().keys()) == set(init_params.keys())


def test_yaml_types():
    d = {
        "tuple": (1, 2, 3),
        "array": np.array([1.0, 42.0]),
        "nested": [{"value": np.float32(0.5)}, np.array([2, 3])],
        "integer": np.int64(4),
    }

    d_out = _make_yaml_friendly(d)
    assert d_out["tuple"] == [1, 2, 3]
    assert d_out["array"] == [1.0, 42.0]
    assert d_out["nested"] == [{"value": 0.5}, [2, 3]]
    assert d_out["integer"] == 4
    assert isinstance(d["nested"][1], np.ndarray)
    assert yaml.safe_load(yaml.safe_dump(d_out)) == d_out
