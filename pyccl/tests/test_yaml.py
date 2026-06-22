import tempfile
import pytest
import filecmp
import io

import numpy as np

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


def test_write_yaml_complex_types():
    cosmo = ccl.CosmologyVanillaLCDM(
        baryonic_effects=ccl.baryons.BaryonsvanDaalen19()
    )
    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
            cosmo.write_yaml(tmpfile)


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
        "array": np.array([1.0, 42.0])
    }

    d_out = _make_yaml_friendly(d)
    assert d_out["tuple"] == [1, 2, 3]
    assert d_out["array"] == [1.0, 42.0]
