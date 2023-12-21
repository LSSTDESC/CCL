import numpy as np
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG

import pytest

# Set tolerances
DISTANCES_TOLERANCE = 1e-4
# The distance tolerance is 1e-3 for distances with massive neutrinos
# This is because we compare to astropy which uses a fitting function
# instead of the full phasespace integral.
# The fitting function itself is not accurate to 1e-4.
DISTANCES_TOLERANCE_MNU = 1e-3

# Tolerance for comparison with CLASS. Currently works up to 1e-6.
DISTANCES_TOLERANCE_CLASS = 1e-5

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.05
h = 0.7
A_s = 2.1e-9
n_s = 0.96
Neff = 0.

# Introduce non-zero values of mu_0 and sigma_0 (mu / Sigma
# parameterisation of modified gravity) for some of the tests
mu_0 = 0.1
sigma_0 = 0.1

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])

mnu = [
    [0.04, 0., 0.], [0.05, 0.01, 0.], [0.05, 0., 0.],
    [0.03, 0.02, 0.]]
# For tests with massive neutrinos, we require N_nu_rel + N_nu_mass = 3
# Because we compare with astropy for benchmarks
# Which assumes N total is split equally among all neutrinos.
Neff_mnu = 3.0


def Neff_from_N_ur_N_ncdm(N_ur, N_ncdm):
    """Calculate N_eff from the number of relativistic
    and massive neutrinos."""
    Neff = N_ur + N_ncdm * ccl.DefaultParams.T_ncdm**4 / (4/11)**(4/3)
    return Neff


class_models = {
    "flat_nonu": {
        "Omega_k": 0.0,
        "Neff": 3.0},
    "pos_curv_nonu": {
        "Omega_k": 0.01,
        "Neff": 3.0},
    "neg_curv_nonu": {
        "Omega_k": -0.01,
        "Neff": 3.0},
    "flat_massnu1": {
        "Omega_k": 0.0,
        # 1 massive neutrino
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=2.0, N_ncdm=1.0),
        "m_nu": [0.0, 0.0, 0.1]},
    "flat_massnu2": {
        "Omega_k": 0.0,
        # 3 massive neutrinos
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=0.0, N_ncdm=3.0),
        "m_nu": [0.03, 0.03, 0.1]},
    "flat_massnu3": {
        "Omega_k": 0.0,
        # 3 massive neutrino
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=0.0, N_ncdm=3.0),
        "m_nu": [0.03, 0.05, 0.1]},
    "flat_manynu1": {
        "Omega_k": 0.0,
        "Neff": 6.0},
    "neg_curv_massnu1": {
        "Omega_k": -0.01,
        # 4 massless, 2 massive neutrino
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=4.0, N_ncdm=2.0),
        "m_nu": [0.0, 0.03, 0.1]},
    "pos_curv_manynu1": {
        "Omega_k": 0.01,
        # 3 massless, 3 massive neutrino
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=3.0, N_ncdm=3.0),
        "m_nu": [0.03, 0.05, 0.1]},
    "CCL1": {
        "Omega_k": 0.0,
        "Neff": 3.046},
    "CCL2": {
        "Omega_k": 0.0,
        "w0": -0.9,
        "wa": 0.0,
        "Neff": 3.046},
    "CCL3": {
        "Omega_k": 0.0,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": 3.046},
    "CCL4": {
        "Omega_k": 0.05,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": 3.046},
    "CCL5": {
        "Omega_k": -0.05,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": 3.046},
    "CCL7": {
        "Omega_k": 0.0,
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=2.0, N_ncdm=1.0),
        "m_nu": [0.04, 0.0, 0.0]},
    "CCL8": {
        "Omega_k": 0.0,
        "w0": -0.9,
        "wa": 0.0,
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=1.0, N_ncdm=2.0),
        "m_nu": [0.05, 0.01, 0.0]},
    "CCL9": {
        "Omega_k": 0.0,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=0.0, N_ncdm=3.0),
        "m_nu": [0.03, 0.02, 0.04]},
    "CCL10": {
        "Omega_k": 0.05,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=2.0, N_ncdm=1.0),
        "m_nu": [0.05, 0.0, 0.0]},
    "CCL11": {
        "Omega_k": -0.05,
        "w0": -0.9,
        "wa": 0.1,
        "Neff": Neff_from_N_ur_N_ncdm(N_ur=1.0, N_ncdm=2.0),
        "m_nu": [0.03, 0.02, 0.0]},
}


def read_chi_benchmark_file():
    """
    Read the file containing all the radial comoving distance benchmarks
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_model1-5.txt").T
    assert (dat.shape == (6, 6))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_chi_hiz_benchmark_file():
    """
    Read the file containing all the radial comoving distance benchmarks
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_hiz_model1-3.txt").T
    assert (dat.shape == (4, 7))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_chi_mnu_benchmark_file():
    """
    Read the file containing all the radial comoving distance benchmarks
    with non-zero massive neutrinos
    (distances are in Mpc)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_mnu_model1-5.txt").T
    assert (dat.shape == (6, 5))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_chi_mnu_hiz_benchmark_file():
    """
    Read the file containing all the radial comoving distance benchmarks
    with non-zero massive neutrinos
    (distances are in Mpc)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_hiz_mnu_model1-5.txt").T
    assert (dat.shape == (6, 7))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_class_allz_chi_benchmark_file():
    """
    Read the file containing the radial comoving distance benchmarks from
    CLASS for the CCL paper models. (distances are in Mpc)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_class_allz.txt").T
    assert (dat.shape == (11, 10))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_class_mnu_chi_benchmark_file():
    """
    Read the file containing the radial comoving distance benchmarks from
    CLASS additional massive neutrino models. (distances are in Mpc)
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/chi_class_extra_mnu.txt").T
    assert (dat.shape == (10, 10))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


def read_dm_benchmark_file():
    """
    Read the file containing all the distance modulus benchmarks
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_model1-5.txt").T
    assert (dat.shape == (6, 6))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


def read_dm_mnu_benchmark_file():
    """
    Read the file containing all the distance modulus benchmarks
    for non-zero massive neutrinos.
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_mnu_model1-5.txt").T
    assert (dat.shape == (6, 5))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


def read_dm_mnu_hiz_benchmark_file():
    """
    Read the file containing all the distance modulus benchmarks
    for non-zero massive neutrinos at high z.
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_hiz_mnu_model1-5.txt").T
    assert (dat.shape == (6, 7))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


def read_class_allz_dm_benchmark_file():
    """
    Read the file containing the distance modulus benchmarks from
    CLASS for the CCL paper models.
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_class_allz.txt").T
    assert (dat.shape == (11, 10))

    # Split into redshift column and dm(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


def read_class_mnu_dm_benchmark_file():
    """
    Read the file containing the distance modulus benchmarks from
    CLASS for additional massive neutrino models.
    """
    # Load data from file
    dat = np.genfromtxt("./benchmarks/data/dm_class_extra_mnu.txt").T
    assert (dat.shape == (10, 10))

    # Split into redshift column and dm(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm


# Set-up test data
z, chi = read_chi_benchmark_file()
zhi, chi_hiz = read_chi_hiz_benchmark_file()
_, dm = read_dm_benchmark_file()
znu, chi_nu = read_chi_mnu_benchmark_file()
znuhi, chi_nu_hiz = read_chi_mnu_hiz_benchmark_file()
z_class_mnu, chi_class_mnu = read_class_mnu_chi_benchmark_file()
z_dm_class_mnu, dm_class_mnu = read_class_mnu_dm_benchmark_file()
z_class_allz, chi_class_allz = read_class_allz_chi_benchmark_file()
z_dm_class_allz, dm_class_allz = read_class_allz_dm_benchmark_file()
_znu, dm_nu = read_dm_mnu_benchmark_file()
_znuhi, dm_nu_hiz = read_dm_mnu_hiz_benchmark_file()


def compare_distances(z, chi_bench, dm_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file.
    This test is only valid when radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0)

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)

    # compare distance moudli where a!=1
    a_not_one = (a != 1).nonzero()
    dm = ccl.distance_modulus(cosmo, a[a_not_one])

    assert np.allclose(
        dm, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE*10)


def compare_distances_hiz(z, chi_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file.
    This test is only valid when radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0)

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)


def compare_distances_mnu(z, chi_bench, dm_bench, Omega_v, w0, wa, Neff, mnu):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff_mnu,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, m_nu=mnu)

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench,
                       atol=1e-12, rtol=DISTANCES_TOLERANCE_MNU)

    # compare distance moudli where a!=1
    a_not_one = (a != 1).nonzero()
    dm = ccl.distance_modulus(cosmo, a[a_not_one])

    assert np.allclose(
        dm, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE_MNU)


def compare_distances_mnu_hiz(z, chi_bench, dm_bench, Omega_v,
                              w0, wa, Neff_mnu, mnu):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, m_nu=mnu)

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench,
                       atol=1e-12, rtol=DISTANCES_TOLERANCE_MNU)

    # compare distance moudli where a!=1
    a_not_one = (a != 1).nonzero()
    dm = ccl.distance_modulus(cosmo, a[a_not_one])

    assert np.allclose(
        dm, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE_MNU)


def compare_class_distances(z, chi_bench, dm_bench, Neff=3.0, m_nu=0.0,
                            Omega_k=0.0, w0=-1.0, wa=0.0):
    """
    Compare distances calculated by pyccl with the distances in the CLASS
    benchmark file.
    """
    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k, m_nu=m_nu,
        w0=w0, wa=wa)

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, rtol=DISTANCES_TOLERANCE_CLASS)

    # Compare distance moudli where a!=1
    a_not_one = a != 1
    dm = ccl.distance_modulus(cosmo, a[a_not_one])
    assert np.allclose(dm, dm_bench[a_not_one], rtol=DISTANCES_TOLERANCE_CLASS)


def compare_distances_muSig(z, chi_bench, dm_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file, for a ccl cosmology with mu / Sigma parameterisation of gravity.
    Nonzero mu / Sigma should NOT affect distances so we compare to the same
    benchmarks as the mu = Sigma = 0 case deliberately.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    # Create new Parameters and Cosmology objects
    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0.,
        mg_parametrization=MuSigmaMG(mu_0=mu_0, sigma_0=sigma_0))

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)

    # compare distance moudli where a!=1
    a_not_one = (a != 1).nonzero()
    dm = ccl.distance_modulus(cosmo, a[a_not_one])

    assert np.allclose(
        dm, dm_bench[a_not_one], atol=1e-3, rtol=DISTANCES_TOLERANCE*10)


def compare_distances_hiz_muSig(z, chi_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark
    file, for a ccl cosmology with mu / Sigma parameterisation of gravity.
    Nonzero mu / Sigma should NOT affect distances so we compare to the same
    benchmarks as the mu = Sigma = 0 case deliberately.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    # Create new Parameters and Cosmology objects
    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
        w0=w0, wa=wa, Omega_g=0.,
        mg_parametrization=MuSigmaMG(mu_0=mu_0, sigma_0=sigma_0))

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert np.allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)


@pytest.mark.parametrize('i', list(range(5)))
def test_distance_model(i):
    compare_distances(
        z, chi[i], dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])


@pytest.mark.parametrize('i', list(range(3)))
def test_distance_hiz_model(i):
    compare_distances_hiz(
        zhi, chi_hiz[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])


@pytest.mark.parametrize('i', list(range(4)))
def test_distance_mnu_model(i):
    compare_distances_mnu(
        znu, chi_nu[i], dm_nu[i], Omega_v_vals[i], w0_vals[i],
        wa_vals[i], Neff, mnu[i])


@pytest.mark.parametrize('i', list(range(4)))
def test_distance_mnu_hiz_model(i):
    compare_distances_mnu(znuhi, chi_nu_hiz[i], dm_nu_hiz[i], Omega_v_vals[i],
                          w0_vals[i], wa_vals[i], Neff, mnu[i])


def test_class_distance_model_flat_nonu():
    i = 0
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["flat_nonu"])


def test_class_distance_model_pos_curv_nonu():
    i = 1
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["pos_curv_nonu"])


def test_class_distance_model_neg_curv_nonu():
    i = 2
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["neg_curv_nonu"])


def test_class_distance_model_flat_massnu1():
    i = 3
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["flat_massnu1"])


def test_class_distance_model_flat_massnu2():
    i = 4
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["flat_massnu2"])


def test_class_distance_model_flat_massnu3():
    i = 5
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["flat_massnu3"])


def test_class_distance_model_flat_manynu1():
    i = 6
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["flat_manynu1"])


def test_class_distance_model_neg_curv_massnu1():
    i = 7
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["neg_curv_massnu1"])


def test_class_distance_model_pos_curv_massnu1():
    i = 8
    compare_class_distances(
        z_class_mnu, chi_class_mnu[i], dm_class_mnu[i],
        **class_models["pos_curv_manynu1"])


def test_class_allz_distance_model_ccl1():
    i = 0
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL1"])


def test_class_allz_distance_model_ccl2():
    i = 1
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL2"])


def test_class_allz_distance_model_ccl3():
    i = 2
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL3"])


def test_class_allz_distance_model_ccl4():
    i = 3
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL4"])


def test_class_allz_distance_model_ccl5():
    i = 4
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL5"])


def test_class_allz_distance_model_ccl7():
    i = 5
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL7"])


def test_class_allz_distance_model_ccl8():
    i = 6
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL8"])


def test_class_allz_distance_model_ccl9():
    i = 7
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL9"])


def test_class_allz_distance_model_ccl10():
    i = 8
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL10"])


def test_class_allz_distance_model_ccl11():
    i = 9
    compare_class_distances(
        z_class_allz, chi_class_allz[i], dm_class_allz[i],
        **class_models["CCL11"])


@pytest.mark.parametrize('i', list(range(5)))
def test_distance_muSig_model(i):
    compare_distances_muSig(
        z, chi[i], dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])


@pytest.mark.parametrize('i', list(range(3)))
def test_distance_hiz_muSig_model(i):
    compare_distances_hiz_muSig(
        zhi, chi_hiz[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])
