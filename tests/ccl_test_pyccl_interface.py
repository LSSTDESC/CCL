import numpy as np
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, run_module_suite
import pyccl as ccl


def reference_models():
    """
    Create a set of reference Cosmology() objects.
    """
    # Standard LCDM model
    p1 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    cosmo1 = ccl.Cosmology(p1)
    
    # LCDM model with curvature
    p2 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, 
                        n_s=0.96, Omega_k=0.05)
    cosmo2 = ccl.Cosmology(p2)
    
    # wCDM model
    p3 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, 
                        n_s=0.96, w0=-0.95, wa=0.05)
    cosmo3 = ccl.Cosmology(p3)
    
    # Return
    return cosmo1, cosmo2, cosmo3

def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.alltrue( np.isfinite(vals) )

def check_background(cosmo):
    """
    Check that background and growth functions can be run.
    """
    # Types of scale factor input (scalar, list, array)
    a_scl = 0.5
    a_lst = [0.2, 0.4, 0.6, 0.8, 1.]
    a_arr = np.linspace(0.2, 1., 5)
    
    # growth_factor
    assert_( all_finite(ccl.growth_factor(cosmo, a_scl)) )
    assert_( all_finite(ccl.growth_factor(cosmo, a_lst)) )
    assert_( all_finite(ccl.growth_factor(cosmo, a_arr)) )
    
    # growth_factor_unnorm
    assert_( all_finite(ccl.growth_factor_unnorm(cosmo, a_scl)) )
    assert_( all_finite(ccl.growth_factor_unnorm(cosmo, a_lst)) )
    assert_( all_finite(ccl.growth_factor_unnorm(cosmo, a_arr)) )
    
    # growth_rate
    assert_( all_finite(ccl.growth_rate(cosmo, a_scl)) )
    assert_( all_finite(ccl.growth_rate(cosmo, a_lst)) )
    assert_( all_finite(ccl.growth_rate(cosmo, a_arr)) )
    
    # comoving_radial_distance
    assert_( all_finite(ccl.comoving_radial_distance(cosmo, a_scl)) )
    assert_( all_finite(ccl.comoving_radial_distance(cosmo, a_lst)) )
    assert_( all_finite(ccl.comoving_radial_distance(cosmo, a_arr)) )
    
    # h_over_h0
    assert_( all_finite(ccl.h_over_h0(cosmo, a_scl)) )
    assert_( all_finite(ccl.h_over_h0(cosmo, a_lst)) )
    assert_( all_finite(ccl.h_over_h0(cosmo, a_arr)) )
    
    # luminosity_distance
    assert_( all_finite(ccl.luminosity_distance(cosmo, a_scl)) )
    assert_( all_finite(ccl.luminosity_distance(cosmo, a_lst)) )
    assert_( all_finite(ccl.luminosity_distance(cosmo, a_arr)) )
    
    # scale_factor_of_chi
    assert_( all_finite(ccl.scale_factor_of_chi(cosmo, a_scl)) )
    assert_( all_finite(ccl.scale_factor_of_chi(cosmo, a_lst)) )
    assert_( all_finite(ccl.scale_factor_of_chi(cosmo, a_arr)) )
    
    # omega_m_z
    assert_( all_finite(ccl.omega_m_z(cosmo, a_scl)) )
    assert_( all_finite(ccl.omega_m_z(cosmo, a_lst)) )
    assert_( all_finite(ccl.omega_m_z(cosmo, a_arr)) )


def check_power(cosmo):
    """
    Check that power spectrum and sigma functions can be run.
    """
    #from power import linear_matter_power, nonlin_matter_power, sigmaR, sigma8
    
    # Types of scale factor
    a = 0.9
    a_arr = np.linspace(0.2, 1., 5.)
    
    # Types of wavenumber input (scalar, list, array)
    k_scl = 1e-1
    k_lst = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    k_arr = np.logspace(-4., 0., 5)
    
    # Types of smoothing scale, R
    R_scl = 8.
    R_lst = [1., 5., 10., 20., 50., 100.]
    R_arr = np.array([1., 5., 10., 20., 50., 100.])
    
    # linear_matter_power
    assert_( all_finite(ccl.linear_matter_power(cosmo, a, k_scl)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, a, k_lst)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, a, k_arr)) )
    
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, a_arr, k_scl)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, a_arr, k_lst)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, a_arr, k_arr)
    
    # nonlin_matter_power
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, a, k_scl)) )
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, a, k_lst)) )
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, a, k_arr)) )
    
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, a_arr, k_scl)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, a_arr, k_lst)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, a_arr, k_arr)
    
    # sigmaR
    assert_( all_finite(ccl.sigmaR(cosmo, R_scl)) )
    assert_( all_finite(ccl.sigmaR(cosmo, R_lst)) )
    assert_( all_finite(ccl.sigmaR(cosmo, R_arr)) )
    
    # sigma8
    assert_( all_finite(ccl.sigma8(cosmo)) )


def check_massfunc(cosmo):
    """
    Check that mass function and supporting functions can be run.
    """
    z = 0.
    z_arr = np.linspace(0., 2., 10)
    mhalo_scl = 1e13
    mhalo_lst = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    mhalo_arr = np.array([1e11, 1e12, 1e13, 1e14, 1e15, 1e16])
    
    # massfunc
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_scl, z)) )
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_lst, z)) )
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_arr, z)) )
    
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_scl, z_arr)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_lst, z_arr)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_arr, z_arr)
    
    # massfunc_m2r
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_scl)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_lst)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_arr)) )
    
    # sigmaM
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_scl, z)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_lst, z)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_arr, z)) )
    
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_scl, z_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_lst, z_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_arr, z_arr)
    

def check_lsst_specs(cosmo):
    """
    Check that lsst_specs functions can be run.
    """
    # Types of scale factor input (scalar, list, array)
    a_scl = 0.5
    a_lst = [0.2, 0.4, 0.6, 0.8, 1.]
    a_arr = np.linspace(0.2, 1., 5)
    
    # Types of redshift input
    z_scl = 0.5
    z_lst = [0., 0.5, 1., 1.5, 2.]
    z_arr = np.array(z_lst)
    
    # p(z) function for dNdz_tomog
    def pz1(z_ph, z_s, args):
        return np.exp(- (z_ph - z_s)**2. / 2.)
    
    # Lambda function p(z) function for dNdz_tomog
    pz2 = lambda z_ph, z_s, args: np.exp(-(z_ph - z_s)**2. / 2.)
    
    # PhotoZFunction classes
    PZ1 = ccl.PhotoZFunction(pz1)
    PZ2 = ccl.PhotoZFunction(pz2)
    
    # bias_clustering
    assert_( all_finite(ccl.bias_clustering(cosmo, a_scl)) )
    assert_( all_finite(ccl.bias_clustering(cosmo, a_lst)) )
    assert_( all_finite(ccl.bias_clustering(cosmo, a_arr)) )
    
    # dNdz_tomog, PhotoZFunction
    # sigmaz_clustering
    assert_( all_finite(ccl.sigmaz_clustering(z_scl)) )
    assert_( all_finite(ccl.sigmaz_clustering(z_lst)) )
    assert_( all_finite(ccl.sigmaz_clustering(z_arr)) )
    
    # sigmaz_sources
    assert_( all_finite(ccl.sigmaz_sources(z_scl)) )
    assert_( all_finite(ccl.sigmaz_sources(z_lst)) )
    assert_( all_finite(ccl.sigmaz_sources(z_arr)) )
    
    # dNdz_tomog
    zmin = 0.
    zmax = 1.
    assert_( all_finite(ccl.dNdz_tomog(z_scl, 'nc', zmin, zmax, PZ1)) )
    assert_( all_finite(ccl.dNdz_tomog(z_lst, 'nc', zmin, zmax, PZ1)) )
    assert_( all_finite(ccl.dNdz_tomog(z_arr, 'nc', zmin, zmax, PZ1)) )
    
    assert_( all_finite(ccl.dNdz_tomog(z_scl, 'nc', zmin, zmax, PZ2)) )
    assert_( all_finite(ccl.dNdz_tomog(z_lst, 'nc', zmin, zmax, PZ2)) )
    assert_( all_finite(ccl.dNdz_tomog(z_arr, 'nc', zmin, zmax, PZ2)) )
    
    assert_( all_finite(ccl.dNdz_tomog(z_scl, 'wl_fid', zmin, zmax, PZ1)) )
    assert_( all_finite(ccl.dNdz_tomog(z_lst, 'wl_fid', zmin, zmax, PZ1)) )
    assert_( all_finite(ccl.dNdz_tomog(z_arr, 'wl_fid', zmin, zmax, PZ1)) )
    
    assert_( all_finite(ccl.dNdz_tomog(z_scl, 'wl_fid', zmin, zmax, PZ2)) )
    assert_( all_finite(ccl.dNdz_tomog(z_lst, 'wl_fid', zmin, zmax, PZ2)) )
    assert_( all_finite(ccl.dNdz_tomog(z_arr, 'wl_fid', zmin, zmax, PZ2)) )
    
    # Argument checking of dNdz_tomog
    # Wrong dNdz_type
    assert_raises(ValueError, ccl.dNdz_tomog, z_scl, 'nonsense', zmin, zmax, PZ1)
    
    # Wrong function type
    assert_raises(TypeError, ccl.dNdz_tomog, z_scl, 'nc', zmin, zmax, pz1)
    assert_raises(TypeError, ccl.dNdz_tomog, z_scl, 'nc', zmin, zmax, z_arr)
    assert_raises(TypeError, ccl.dNdz_tomog, z_scl, 'nc', zmin, zmax, None)
    

def test_background():
    """
    Test background and growth functions in ccl.background.
    """
    for cosmo in reference_models():
        yield check_background, cosmo

def test_power():
    """
    Test power spectrum and sigma functions in ccl.power.
    """
    for cosmo in reference_models():
        pass # FIXME yield check_power, cosmo

def test_massfunc():
    """
    Test mass function and supporting functions.
    """
    for cosmo in reference_models():
        pass #yield check_massfunc, cosmo

def test_lsst_specs():
    """
    Test lsst specs.
    """
    for cosmo in reference_models():
        yield check_lsst_specs, cosmo

"""
# Background cosmology functions and growth functions
from background import growth_factor, growth_factor_unnorm, growth_rate, \
                       comoving_radial_distance, h_over_h0, \
                       luminosity_distance, scale_factor_of_chi, omega_m_z

# Power spectrum calculations and sigma8
from power import linear_matter_power, nonlin_matter_power, sigmaR, sigma8

# Halo mass function
from massfunction import massfunc, massfunc_m2r, sigmaM

# Cl's and tracers
from cls import angular_cl, ClTracer, ClTracerNumberCounts, ClTracerLensing

from lsst_specs import bias_clustering, sigmaz_clustering, sigmaz_sources, \
                       dNdz_tomog, PhotoZFunction

# Useful constants and unit conversions
from constants import CLIGHT_HMPC, MPC_TO_METER, PC_TO_METER, \
                      GNEWT, RHO_CRITICAL, SOLAR_MASS
"""

if __name__ == '__main__':
    run_module_suite()
