import numpy as np
from numpy.testing import assert_allclose, run_module_suite
import pyccl as ccl
from os.path import dirname,join
# Set tolerances
GROWTH_TOLERANCE = 1e-4

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.05
N_nu_rel = 0.
N_nu_mass = 0.
m_nu=0.
h = 0.7
A_s = 2.1e-9
n_s = 0.96

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1])

def read_growth_test_file():
    """
    Read the file containing all the radial comoving distance benchmarks 
    (distances are in Mpc/h)
    """
    # Load data from file
    dat = np.genfromtxt(join(dirname(__file__),"benchmark/growth_model1-5.txt")).T
    assert(dat.shape == (6,6))
    
    # Split into redshift column and growth(z) columns
    z = dat[0]
    gfac = dat[1:]
    return z, gfac

# Set-up test data
z, gfac = read_growth_test_file()

def compare_growth(z, gfac_bench, Omega_v, w0, wa):
    """
    Compare growth factor calculated by pyccl with the values in the benchmark 
    file. This test only works if radiation is explicitly set to 0.
    """
    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v    
    
    # Create new Parameters and Cosmology objects
    p = ccl.Parameters(Omega_c=Omega_c, Omega_b=Omega_b, N_nu_rel=N_nu_rel, N_nu_mass=N_nu_mass, m_nu=m_nu,
                       h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k,
                       w0=w0, wa=wa)
    p.parameters.Omega_g = 0. # Hack to set to same value used for benchmarks
    cosmo = ccl.Cosmology(p)
    
    # Calculate distance using pyccl
    a = 1. / (1. + z)
    gfac = ccl.growth_factor_unnorm(cosmo, a)
    
    # Compare to benchmark data
    assert_allclose(gfac, gfac_bench, atol=1e-12, rtol=GROWTH_TOLERANCE)


def test_growth_model_0():
    i = 0
    compare_growth(z, gfac[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_growth_model_1():
    i = 1
    compare_growth(z, gfac[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_growth_model_2():
    i = 2
    compare_growth(z, gfac[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_growth_model_3():
    i = 3
    compare_growth(z, gfac[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])

def test_growth_model_4():
    i = 4
    compare_growth(z, gfac[i], Omega_v_vals[i], w0_vals[i], wa_vals[i])


def test_mgrowth():
    """
    Compare the modified growth function computed by CCL against the exact 
    result for a particular modification of the growth rate.
    """
    # Define differential growth rate arrays
    nz_mg = 128
    z_mg = np.zeros(nz_mg)
    df_mg = np.zeros(nz_mg)
    for i in range(0, nz_mg):
        z_mg[i] = 4. * (i + 0.0) / (nz_mg - 1.)
        df_mg[i] = 0.1 / (1. + z_mg[i])
    
    # Define two test cosmologies, without and with modified growth respectively
    p1 = ccl.Parameters(Omega_c=0.25, Omega_b=0.05, Omega_k=0., N_nu_rel=0., N_nu_mass=0., m_nu=0., 
                        w0=-1., wa=0., h=0.7, A_s=2.1e-9, n_s=0.96)
    p2 = ccl.Parameters(Omega_c=0.25, Omega_b=0.05, Omega_k=0., N_nu_rel=0., N_nu_mass=0., m_nu=0.,  
                        w0=-1., wa=0., h=0.7, A_s=2.1e-9, n_s=0.96, 
                        z_mg=z_mg, df_mg=df_mg)
    cosmo1 = ccl.Cosmology(p1)
    cosmo2 = ccl.Cosmology(p2)
    
    # We have included a growth modification \delta f = K*a, with K==0.1 
    # (arbitrarily). This case has an analytic solution, given by 
    # D(a) = D_0(a)*exp(K*(a-1)). Here we compare the growth computed by CCL
    # with the analytic solution.
    a = 1. / (1. + z_mg)
    
    d1 = ccl.growth_factor(cosmo1, a)
    d2 = ccl.growth_factor(cosmo2, a)
    f1 = ccl.growth_rate(cosmo1, a)
    f2 = ccl.growth_rate(cosmo2, a)
    
    f2r = f1 + 0.1*a
    d2r = d1 * np.exp(0.1*(a-1.))
    
    # Check that ratio of calculated and analytic results is within tolerance
    assert_allclose(d2r/d2, np.ones(d2.size), rtol=GROWTH_TOLERANCE)
    assert_allclose(f2r/f2, np.ones(f2.size), rtol=GROWTH_TOLERANCE)


if __name__ == "__main__":
    run_module_suite()
