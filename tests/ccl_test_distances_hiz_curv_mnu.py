import numpy as np
from numpy.testing import assert_allclose, run_module_suite, decorators
import pyccl as ccl
from os.path import dirname,join,abspath
from collections import OrderedDict

# Set tolerances
DISTANCES_TOLERANCE = 1e-5

# Set up the cosmological parameters to be used in each of the models
Omega_c = 0.25
Omega_b = 0.05
h = 0.7
A_s = 2.1e-9
n_s = 0.96

def Neff(N_ur, N_ncdm):
    Neff = N_ur + N_ncdm * ccl.ccllib.TNCDM**4 / (4./11.)**(4./3.)
    return Neff

models = OrderedDict(
            {"flat_nonu"       : {"Omega_k"  : 0.0,
                                  "Neff"     : 3.0},
            "pos_curv_nonu"    : {"Omega_k"  : 0.01,
                                  "Neff"     : 3.0},
            "neg_curv_nonu"    : {"Omega_k"  : -0.01,
                                  "Neff"     : 3.0},
            "flat_massnu1"     : {"Omega_k"  : 0.0,
                                  "Neff"     : Neff(N_ur=2.0, N_ncdm=1.0),  # 1 massive neutrino
                                  "m_nu"     : [0.0, 0.0, 0.1]},            # Mass
            "flat_massnu2"     : {"Omega_k"  : 0.0,
                                  "Neff"     : Neff(N_ur=0.0, N_ncdm=3.0),   # 3 massive neutrino
                                  "m_nu"     : [0.03, 0.03, 0.1]},           # Masses
            "flat_massnu3"     : {"Omega_k"  : 0.0,
                                  "Neff"     : Neff(N_ur=0.0, N_ncdm=3.0),   # 3 massive neutrino
                                  "m_nu"     : [0.03, 0.05, 0.1]}, # Masses
            "flat_manynu1"     : {"Omega_k"  : 0.0,
                                  "Neff"     : 6.0},               # 6 massless neutrinos
            "neg_curv_massnu1" : {"Omega_k"  : -0.01,
                                  "Neff"     : Neff(N_ur=4.0, N_ncdm=2.0),   # 4 massless, 2 massive neutrino
                                  "m_nu"     : [0.0, 0.03, 0.1]}, # Masses
            "pos_curv_manynu1" : {"Omega_k"  : 0.01,
                                  "Neff"     : Neff(N_ur=3.0, N_ncdm=3.0),   # 3 massless, 3 massive neutrino
                                  "m_nu"     : [0.03, 0.05, 0.1]}, # Masses
            }
        )


path = dirname(abspath(__file__))
def read_chi_test_file():
    """
    Read the file containing all the radial comoving distance benchmarks 
    (distances are in Mpc)
    """
    # Load data from file
    dat = np.genfromtxt(join(path, "benchmark/chi_hiz_mnu_model6-15.txt")).T
    assert(dat.shape == (10, 10))
    
    # Split into redshift column and chi(z) columns
    z = dat[0]
    chi = dat[1:]
    return z, chi


# Set-up test data
z, chi = read_chi_test_file()

#@decorators.slow    
def compare_distances_mnu_curv(z, chi_bench, Neff=3.0, m_nu=0.0, Omega_k=0.0):
    """
    Compare distances calculated by pyccl with the distances in the benchmark 
    file.
    """
    # Create new Parameters and Cosmology objects
    p = ccl.Parameters(Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff, 
                       h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k, m_nu=m_nu)
    cosmo = ccl.Cosmology(p)
    
    # Calculate distance using pyccl
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    # Compare to benchmark data
    assert_allclose(chi, chi_bench, rtol=DISTANCES_TOLERANCE)


def test_distance_model_flat_nonu():
    i = 0
    compare_distances_mnu_curv(z, chi[i], **models["flat_nonu"])

def test_distance_model_pos_curv_nonu():
    i = 1
    compare_distances_mnu_curv(z, chi[i], **models["pos_curv_nonu"])

def test_distance_model_neg_curv_nonu():
    i = 2
    compare_distances_mnu_curv(z, chi[i], **models["neg_curv_nonu"])

def test_distance_model_flat_massnu1():
    i = 3
    compare_distances_mnu_curv(z, chi[i], **models["flat_massnu1"])

def test_distance_model_flat_massnu2():
    i = 4
    compare_distances_mnu_curv(z, chi[i], **models["flat_massnu2"])

def test_distance_model_flat_massnu3():
    i = 5
    compare_distances_mnu_curv(z, chi[i], **models["flat_massnu3"])

def test_distance_model_flat_manynu1():
    i = 6
    compare_distances_mnu_curv(z, chi[i], **models["flat_manynu1"])

def test_distance_model_neg_curv_massnu1():
    i = 7
    compare_distances_mnu_curv(z, chi[i], **models["neg_curv_massnu1"])

def test_distance_model_pos_curv_massnu1():
    i = 8
    compare_distances_mnu_curv(z, chi[i], **models["pos_curv_manynu1"])

if __name__ == "__main__":
    run_module_suite()
