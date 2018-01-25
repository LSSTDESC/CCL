import numpy as np
from numpy.testing import assert_allclose, run_module_suite
import pyccl as ccl
from os.path import dirname,join,abspath
# Set tolerances
DISTANCES_TOLERANCE = 1e-4

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.05
N_nu_rel = 0.
h = 0.7
A_s = 2.1e-9
n_s = 0.96

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])
path = dirname(abspath(__file__))
def read_chi_test_file():
    """
    Read the file containing all the radial comoving distance benchmarks 
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt(join(path,"benchmark/chi_model1-5.txt")).T
    assert(dat.shape == (6,6))
    
    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi

def read_chi_hiz_test_file():
    """
    Read the file containing all the radial comoving distance benchmarks 
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt(join(path,"benchmark/chi_hiz_model1-3.txt")).T
    assert(dat.shape == (4,7))
    
    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi

def read_dm_test_file():
    """
    Read the file containing all the distance modulus benchmarks 
    """
    # Load data from file
    dat = np.genfromtxt(join(path,"benchmark/dm_model1-5.txt")).T
    assert(dat.shape == (6,6))

    # Split into redshift column and chi(z) columns
    z = dat[0]
    dm = dat[1:]
    return z, dm

# Set-up test data
z, chi = read_chi_test_file()
zhi, chi_hiz = read_chi_hiz_test_file()
_, dm = read_dm_test_file()

def compare_distances(z, chi_bench,dm_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark 
    file.
    This test is only valid when radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v    
    
    # Create new Parameters and Cosmology objects
    p = ccl.Parameters(Omega_c=Omega_c, Omega_b=Omega_b, N_nu_rel=N_nu_rel, 
                       h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
                       w0=w0, wa=wa)
    p.parameters.Omega_g = 0. # Hack to set to same value used for benchmarks
    cosmo = ccl.Cosmology(p)
    
    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert_allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)

    #compare distance moudli where a!=1
    a_not_one = (a!=1).nonzero()
    dm = ccl.distance_modulus(cosmo,a[a_not_one])

    assert_allclose(dm, dm_bench[a_not_one], atol=1e-3, rtol = DISTANCES_TOLERANCE*10)

def compare_distances_hiz(z, chi_bench, Omega_v, w0, wa):
    """
    Compare distances calculated by pyccl with the distances in the benchmark 
    file.
    This test is only valid when radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v    
    
    # Create new Parameters and Cosmology objects
    p = ccl.Parameters(Omega_c=Omega_c, Omega_b=Omega_b, N_nu_rel=N_nu_rel, 
                       h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
                       w0=w0, wa=wa)
    p.parameters.Omega_g = 0. # Hack to set to same value used for benchmarks
    cosmo = ccl.Cosmology(p)
    
    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a) * h
    # Compare to benchmark data
    assert_allclose(chi, chi_bench, atol=1e-12, rtol=DISTANCES_TOLERANCE)
    
def test_distance_model_0():
    i = 0
    compare_distances(z, chi[i],dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_model_1():
    i = 1
    compare_distances(z, chi[i],dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_model_2():
    i = 2
    compare_distances(z, chi[i],dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_model_3():
    i = 3
    compare_distances(z, chi[i],dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_model_4():
    i = 4
    compare_distances(z, chi[i],dm[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_hiz_model_0():
    i = 0
    compare_distances_hiz(zhi, chi_hiz[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_hiz_model_1():
    i = 1
    compare_distances_hiz(zhi, chi_hiz[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_distance_hiz_model_2():
    i = 2
    compare_distances_hiz(zhi, chi_hiz[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])
    

if __name__ == "__main__":
    run_module_suite()
