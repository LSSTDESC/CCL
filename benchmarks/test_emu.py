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

    mnu = ccl.nu_masses(m_total=cosmos[model, 7]*cosmos[model, 2]**2,
                        mass_split='equal')

    cosmo = ccl.Cosmology(
        Omega_c=cosmos[model, 0],
        Omega_b=cosmos[model, 1],
        h=cosmos[model, 2],
        sigma8=cosmos[model, 3],
        n_s=cosmos[model, 4],
        w0=cosmos[model, 5],
        wa=cosmos[model, 6],
        m_nu=mnu,
        mass_split='list',
        Neff=3.04,
        Omega_g=0,
        Omega_k=0,
        transfer_function='boltzmann_camb',
        matter_power_spectrum='emu',
    )

    a = 1
    k = data[:, 0]

    pk = ccl.nonlin_matter_power(cosmo, k, a)

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


@pytest.mark.parametrize('model', list(range(3)))
def test_emu_lin(model):
    cosmos = np.loadtxt("./benchmarks/data/emu_input_cosmologies.txt")

    mnu = ccl.nu_masses(m_total=cosmos[model, 7]*cosmos[model, 2]**2,
                        mass_split='equal')

    cosmo = ccl.Cosmology(
        Omega_c=cosmos[model, 0],
        Omega_b=cosmos[model, 1],
        h=cosmos[model, 2],
        sigma8=cosmos[model, 3],
        n_s=cosmos[model, 4],
        w0=cosmos[model, 5],
        wa=cosmos[model, 6],
        m_nu=mnu,
        mass_split='list',
        Neff=3.04,
        Omega_g=0,
        Omega_k=0,
        transfer_function='boltzmann_camb',
        matter_power_spectrum='emu',
    )

    a = 1
    k = np.logspace(-3, -2, 50)

    pk = ccl.nonlin_matter_power(cosmo, k, a)
    pk_lin = ccl.linear_matter_power(cosmo, k, a)

    err = np.abs(pk/pk_lin-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)
