import numpy as np
import pytest

import pyccl as ccl

EMU_TOLERANCE = 3.0E-2


@pytest.mark.parametrize('model', list(range(4)))
def test_emu_nu(model):
    model_numbers = [38, 39, 40, 42]
    data = np.loadtxt(
        "./benchmarks/data/emu_nu_smooth_pk_M%d.txt" % model_numbers[model])

    cosmos = np.loadtxt("./benchmarks/data/emu_nu_cosmologies.txt")

    mnu = ccl.nu_masses(
        cosmos[model, 7] * cosmos[model, 2]**2, 'equal', T_CMB=2.725)

    cosmo = ccl.Cosmology(
        Omega_c=cosmos[model, 0],
        Omega_b=cosmos[model, 1],
        h=cosmos[model, 2],
        sigma8=cosmos[model, 3],
        n_s=cosmos[model, 4],
        w0=cosmos[model, 5],
        wa=cosmos[model, 6],
        m_nu=mnu,
        m_nu_type='list',
        Neff=3.04,
        Omega_g=0,
        Omega_k=0,
        transfer_function='boltzmann_class',
        matter_power_spectrum='emu',
    )

    a = 1
    k = data[:, 0]

    # Catch warning about neutrino linear growth
    pk = ccl.pyutils.assert_warns(ccl.CCLWarning,
                                  ccl.nonlin_matter_power,
                                  cosmo, k, a)
    err = np.abs(pk/data[:, 1]-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)


@pytest.mark.parametrize('model', [0, 1, 3, 5])
def test_emu(model):
    model_numbers = [1, 3, 5, 6, 8, 10]
    data = np.loadtxt(
        "./benchmarks/data/emu_smooth_pk_M%d.txt" % model_numbers[model])

    cosmos = np.loadtxt("./benchmarks/data/emu_cosmologies.txt")

    cosmo = ccl.Cosmology(
        Omega_c=cosmos[model, 0],
        Omega_b=cosmos[model, 1],
        h=cosmos[model, 2],
        sigma8=cosmos[model, 3],
        n_s=cosmos[model, 4],
        w0=cosmos[model, 5],
        wa=cosmos[model, 6],
        Neff=3.04,
        Omega_g=0,
        Omega_k=0,
        transfer_function='bbks',
        matter_power_spectrum='emu',
    )

    a = 1
    k = data[:, 0]
    pk = ccl.nonlin_matter_power(cosmo, k, a)
    err = np.abs(pk/data[:, 1]-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)
