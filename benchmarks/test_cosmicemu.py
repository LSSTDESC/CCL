import numpy as np
import pyccl as ccl
import pytest

CEMU_TOL = 1E-3
# Old emulator tolerance
EMU_TOLERANCE = 3.0E-2
cemu_old = ccl.CosmicemuMTIIPk('tot')


@pytest.mark.parametrize("kind", ["tot", "cb"])
def test_cemu_mtiv(kind):
    cemu = ccl.CosmicemuMTIVPk(kind)
    for i, omega_nu in enumerate([0.0, 1E-3]):
        m_nu = ccl.nu_masses(Omega_nu_h2=omega_nu, mass_split='equal')
        h = 0.67
        cosmo = ccl.Cosmology(
            Omega_c=0.3-0.05-omega_nu/h**2,
            Omega_b=0.05,
            h=h,
            sigma8=0.8,
            n_s=0.96,
            m_nu=m_nu,
            mass_split='equal',
            w0=-1.0,
            wa=0.0,
            matter_power_spectrum=cemu)

        zs = np.array([0.0, 1.0])
        for iz, a in enumerate(1/(1+zs)):
            k, pk = np.loadtxt(
                f"./benchmarks/data/cosmo{i+1}_{kind}_{iz}.txt",
                unpack=True)
            pkh = cosmo.nonlin_matter_power(k, a)
            assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)

            ktest, pktest = cemu.get_pk_at_a(cosmo, a)
            pkh = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
            assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)


@pytest.mark.parametrize("kind", ["tot", "cb"])
def test_cemu_mtii(kind):
    cemu = ccl.CosmicemuMTIIPk(kind)
    for i, omega_nu in enumerate([0.0, 1E-3]):
        m_nu = ccl.nu_masses(Omega_nu_h2=omega_nu, mass_split='equal')
        h = 0.67
        cosmo = ccl.Cosmology(
            Omega_c=0.3-0.05-omega_nu/h**2,
            Omega_b=0.05,
            h=h,
            sigma8=0.8,
            n_s=0.96,
            m_nu=m_nu,
            mass_split='equal',
            w0=-1.0,
            wa=0.0,
            matter_power_spectrum=cemu)

        zs = np.array([0.0, 1.0])
        for iz, a in enumerate(1/(1+zs)):
            k, pk = np.loadtxt(
                f"./benchmarks/data/cosmo{i+1}_MTII_{kind}_{iz}.txt",
                unpack=True)
            pkh = cosmo.nonlin_matter_power(k, a)
            assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)

            ktest, pktest = cemu.get_pk_at_a(cosmo, a)
            pkh = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
            assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)


@pytest.mark.parametrize('model', list(range(4)))
def test_emu_nu_old(model):
    model_numbers = [38, 39, 40, 42]
    data = np.loadtxt(
        "./benchmarks/data/emu_nu_smooth_pk_M%d.txt" % model_numbers[model])

    cosmos = np.loadtxt("./benchmarks/data/emu_nu_cosmologies.txt")

    mnu = ccl.nu_masses(Omega_nu_h2=cosmos[model, 7]*cosmos[model, 2]**2,
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
        matter_power_spectrum=cemu_old,
    )

    a = 1
    k = data[:, 0]

    pk = ccl.nonlin_matter_power(cosmo, k, a)

    err = np.abs(pk/data[:, 1]-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)


@pytest.mark.parametrize('model', [0, 1, 3, 5])
def test_emu_old(model):
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
        matter_power_spectrum=cemu_old,
    )

    a = 1
    k = data[:, 0]
    pk = ccl.nonlin_matter_power(cosmo, k, a)
    err = np.abs(pk/data[:, 1]-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)


@pytest.mark.parametrize('model', list(range(3)))
def test_emu_lin_old(model):
    cosmos = np.loadtxt("./benchmarks/data/emu_input_cosmologies.txt")

    mnu = ccl.nu_masses(Omega_nu_h2=cosmos[model, 7]*cosmos[model, 2]**2,
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
        matter_power_spectrum=cemu_old,
    )

    a = 1
    k = np.logspace(-3, -2, 50)

    pk = ccl.nonlin_matter_power(cosmo, k, a)
    pk_lin = ccl.linear_matter_power(cosmo, k, a)

    err = np.abs(pk/pk_lin-1)
    assert np.allclose(err, 0, rtol=0, atol=EMU_TOLERANCE)
