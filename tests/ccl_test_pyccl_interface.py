import numpy as np,math
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, decorators, run_module_suite, assert_allclose
import pyccl as ccl
from pyccl import CCLError

def reference_models():
    """
    Create a set of reference Cosmology() objects.
    """
    # Standard LCDM model
    cosmo1 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

    # LCDM model with curvature
    cosmo2 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10,
        n_s=0.96, Omega_k=0.05)

    # wCDM model
    cosmo3 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10,
        n_s=0.96, w0=-0.95, wa=0.05)

    # BBKS Pk
    cosmo4 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks',
        matter_power_spectrum='linear')

    # E&H Pk
    cosmo5 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear')

    # Emulator Pk
    cosmo6 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
        n_s=0.96, Neff=3.04, m_nu=0.,
        transfer_function='boltzmann_class',
        matter_power_spectrum='emu')

    # Baryons Pk
    cosmo8 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96,
        baryons_power_spectrum='bcm')

    # Baryons Pk with choice of BCM parameters other than default
    cosmo9 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96,
        bcm_log10Mc=math.log10(1.7e14), bcm_etab=0.3, bcm_ks=75.,
        baryons_power_spectrum='bcm')

    # Emulator Pk w/neutrinos force equalize
    #p10 = ccl.Parameters(Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
    #                    n_s=0.96, Neff=3.04, m_nu=[0.02, 0.02, 0.02])
    #cosmo10 = ccl.Cosmology(p7, transfer_function='emulator',
    #                       matter_power_spectrum='emu', emulator_neutrinos='equalize')

    # Return (do a few cosmologies, for speed reasons)
    return [cosmo1, cosmo4, cosmo5, cosmo9] # cosmo2, cosmo3, cosmo6

def reference_models_nu():
    """
    Create a set of reference cosmological models with massive neutrinos.
    This is separate because certain functionality is not yes implemented
    for massive neutrino cosmologies so will throw errors.
    """

    # Emulator Pk w/neutrinos list
    cosmo1 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
        n_s=0.96, Neff=3.04, m_nu=[0.02, 0.02, 0.02],
        transfer_function='boltzmann_class',
        matter_power_spectrum='emu')

    # Emulator Pk with neutrinos, force equalize
    #p2 = ccl.Parameters(Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
    #                    n_s=0.96, Neff=3.04, m_nu=0.11)
    #cosmo2 = ccl.Cosmology(p1, transfer_function='emulator', 
    #                       matter_power_spectrum='emu', emulator_neutrinos='equalize') 
    
    return [cosmo1]  
    
def reference_models_mg():
    """ Create a set of reference cosmological models with the mu / Sigma
    parameterisation of modified gravity. """
	
	# Ask for linear matter power spectrum because that is what is implemented
	# for mu / Sigma.                                       
    cosmo1 = ccl.Cosmology(Omega_c = 0.27, Omega_b = 0.022/0.67**2, h = 0.67, sigma8 = 0.8,
                         n_s = 0.96, mu_0 = 0.1, sigma_0 = 0.1, matter_power_spectrum='linear')
    # And ask for halofit to make sure we can throw an error if this is called.
    cosmo2 = ccl.Cosmology(Omega_c = 0.27, Omega_b = 0.022/0.67**2, h = 0.67, sigma8 = 0.8,
                         n_s = 0.96, mu_0 = 0.1, sigma_0 = 0.1, matter_power_spectrum='halofit')
    
    return [cosmo1]

def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.all( np.isfinite(vals) )

def check_background(cosmo):
    """
    Check that background and growth functions can be run.
    """

    # Types of scale factor input (scalar, list, array)
    a_scl = 0.5
    is_comoving = 0
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

    # comoving_angular_distance
    assert_( all_finite(ccl.comoving_angular_distance(cosmo, a_scl)) )
    assert_( all_finite(ccl.comoving_angular_distance(cosmo, a_lst)) )
    assert_( all_finite(ccl.comoving_angular_distance(cosmo, a_arr)) )

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

    # omega_m_a
    assert_( all_finite(ccl.omega_x(cosmo, a_scl, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_lst, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'matter')) )

    # Fractional density of different types of fluid
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'dark_energy')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'radiation')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'curvature')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'neutrinos_rel')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'neutrinos_massive')) )

    # Check that omega_x fails if invalid component type is passed
    assert_raises(ValueError, ccl.omega_x, cosmo, a_scl, 'xyz')

    # rho_crit_a
    assert_( all_finite(ccl.rho_x(cosmo, a_scl, 'critical', is_comoving)) )
    assert_( all_finite(ccl.rho_x(cosmo, a_lst, 'critical', is_comoving)) )
    assert_( all_finite(ccl.rho_x(cosmo, a_arr, 'critical', is_comoving)) )

    # rho_m_a
    assert_( all_finite(ccl.rho_x(cosmo, a_scl, 'matter', is_comoving)) )
    assert_( all_finite(ccl.rho_x(cosmo, a_lst, 'matter', is_comoving)) )
    assert_( all_finite(ccl.rho_x(cosmo, a_arr, 'matter', is_comoving)) )
    
    # mu 
    assert_( all_finite(ccl.background.mu_MG(cosmo, a_scl)) )
    assert_( all_finite(ccl.mu_MG(cosmo, a_lst)) )
    assert_( all_finite(ccl.mu_MG(cosmo, a_arr)) )
    
    # Sig
    assert_( all_finite(ccl.Sig_MG(cosmo, a_scl)) )
    assert_( all_finite(ccl.Sig_MG(cosmo, a_lst)) )
    assert_( all_finite(ccl.Sig_MG(cosmo, a_arr)) )


def check_background_nu(cosmo):
    """
    Check that background functions can be run and that the growth functions
    exit gracefully in functions with massive neutrinos (not implemented yet).
    """
    # Types of scale factor input (scalar, list, array)
    a_scl = 0.5
    a_lst = [0.2, 0.4, 0.6, 0.8, 1.]
    a_arr = np.linspace(0.2, 1., 5)

    # growth_factor
    assert_raises(CCLError, ccl.growth_factor, cosmo, a_scl)
    assert_raises(CCLError, ccl.growth_factor, cosmo, a_lst)
    assert_raises(CCLError, ccl.growth_factor, cosmo, a_arr)

    # growth_factor_unnorm
    assert_raises(CCLError, ccl.growth_factor_unnorm, cosmo, a_scl)
    assert_raises(CCLError, ccl.growth_factor_unnorm, cosmo, a_lst)
    assert_raises(CCLError, ccl.growth_factor_unnorm, cosmo, a_arr)

    # growth_rate
    assert_raises(CCLError,ccl.growth_rate, cosmo, a_scl)
    assert_raises(CCLError,ccl.growth_rate, cosmo, a_lst)
    assert_raises(CCLError,ccl.growth_rate, cosmo, a_arr)

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

    # omega_m_a
    assert_( all_finite(ccl.omega_x(cosmo, a_scl, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_lst, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'matter')) )
    
    # mu 
    assert_( all_finite(ccl.mu_MG(cosmo, a_scl)) )
    assert_( all_finite(ccl.mu_MG(cosmo, a_lst)) )
    assert_( all_finite(ccl.mu_MG(cosmo, a_arr)) )
    
    # Sig
    assert_( all_finite(ccl.Sig_MG(cosmo, a_scl)) )
    assert_( all_finite(ccl.Sig_MG(cosmo, a_lst)) )
    assert_( all_finite(ccl.Sig_MG(cosmo, a_arr)) )
   
def check_power(cosmo):
    """
    Check that power spectrum and sigma functions can be run.
    """
    # Types of scale factor
    a = 0.9
    a_arr = np.linspace(0.2, 1., 5)

    # Types of wavenumber input (scalar, list, array)
    k_scl = 1e-1
    k_lst = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    k_arr = np.logspace(-4., 0., 5)

    # Types of smoothing scale, R
    R_scl = 8.
    R_lst = [1., 5., 10., 20., 50., 100.]
    R_arr = np.array([1., 5., 10., 20., 50., 100.])

    # linear_matter_power
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_scl, a)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_lst, a)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_arr, a)) )

    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_scl, a_arr)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_lst, a_arr)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_arr, a_arr)

    # nonlin_matter_power
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_scl, a)) )
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_lst, a)) )
    assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_arr, a)) )

    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_scl, a_arr)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_lst, a_arr)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_arr, a_arr)

    # sigmaR
    assert_( all_finite(ccl.sigmaR(cosmo, R_scl)) )
    assert_( all_finite(ccl.sigmaR(cosmo, R_lst)) )
    assert_( all_finite(ccl.sigmaR(cosmo, R_arr)) )

    # sigmaV
    assert_( all_finite(ccl.sigmaV(cosmo, R_scl)) )
    assert_( all_finite(ccl.sigmaV(cosmo, R_lst)) )
    assert_( all_finite(ccl.sigmaV(cosmo, R_arr)) )

    # sigma8
    assert_( all_finite(ccl.sigma8(cosmo)) )
    
def check_power_MG(cosmo):
    """
    Check that power spectrum and sigma functions can be run.
    """
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
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_scl, a)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_lst, a)) )
    assert_( all_finite(ccl.linear_matter_power(cosmo, k_arr, a)) )
    
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_scl, a_arr)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_lst, a_arr)
    assert_raises(TypeError, ccl.linear_matter_power, cosmo, k_arr, a_arr)
    
    # nonlin_matter_power
    ### Changes mean that this check of whether the power spectrum tag is 
    ### linear or not doesn't work anymore. Need to fix this. 
    """if (cosmo.__getitem__(matter_power_spectrum)== 'linear'):
        assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_scl, a)) )
        assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_lst, a)) )
        assert_( all_finite(ccl.nonlin_matter_power(cosmo, k_arr, a)) )
    else:
        print cosmo.__getitem__(matter_power_spectrum)
        assert_raises(RuntimeError, ccl.nonlin_matter_power, cosmo, k_scl, a)
        assert_raises(RuntimeError, ccl.nonlin_matter_power, cosmo, k_lst, a)
        assert_raises(RuntimeError, ccl.nonlin_matter_power, cosmo, k_arr, a)"""
    
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_scl, a_arr)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_lst, a_arr)
    assert_raises(TypeError, ccl.nonlin_matter_power, cosmo, k_arr, a_arr)

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
    a = 1.
    a_arr = 1. / (1.+z_arr)
    mhalo_scl = 1e13
    mhalo_lst = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    mhalo_arr = np.array([1e11, 1e12, 1e13, 1e14, 1e15, 1e16])
    odelta = 200.

    # massfunc
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_scl, a, odelta)) )
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_lst, a, odelta)) )
    assert_( all_finite(ccl.massfunc(cosmo, mhalo_arr, a, odelta)) )

    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_scl, a_arr, odelta)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_lst, a_arr, odelta)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_arr, a_arr, odelta)

    # Check whether odelta out of bounds
    assert_raises(CCLError, ccl.massfunc, cosmo, mhalo_scl, a, 199.)
    assert_raises(CCLError, ccl.massfunc, cosmo, mhalo_scl, a, 5000.)

    # massfunc_m2r
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_scl)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_lst)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_arr)) )

    # sigmaM
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_scl, a)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_lst, a)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_arr, a)) )

    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_scl, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_lst, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_arr, a_arr)

    # halo_bias
    assert_( all_finite(ccl.halo_bias(cosmo, mhalo_scl, a)) )
    assert_( all_finite(ccl.halo_bias(cosmo, mhalo_lst, a)) )
    assert_( all_finite(ccl.halo_bias(cosmo, mhalo_arr, a)) )


def check_massfunc_nu(cosmo):
    """
    Check that mass function and supporting functions can be run.
    """
    z = 0.
    z_arr = np.linspace(0., 2., 10)
    a = 1.
    a_arr = 1. / (1.+z_arr)
    mhalo_scl = 1e13
    mhalo_lst = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    mhalo_arr = np.array([1e11, 1e12, 1e13, 1e14, 1e15, 1e16])
    odelta = 200.

    # massfunc
    assert_raises(CCLError, ccl.massfunc,cosmo, mhalo_scl, a, odelta)
    assert_raises(CCLError, ccl.massfunc,cosmo, mhalo_lst, a, odelta)
    assert_raises(CCLError, ccl.massfunc,cosmo, mhalo_arr, a, odelta)

    # halo bias
    assert_raises(CCLError, ccl.halo_bias, cosmo, mhalo_scl, a, odelta)
    assert_raises(CCLError, ccl.halo_bias, cosmo, mhalo_lst, a, odelta)
    assert_raises(CCLError, ccl.halo_bias, cosmo, mhalo_arr, a, odelta)

    # massfunc_m2r
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_scl)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_lst)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_arr)) )

    # sigmaM
    assert_raises(CCLError, ccl.sigmaM, cosmo, mhalo_scl, a)
    assert_raises(CCLError, ccl.sigmaM, cosmo, mhalo_lst, a)
    assert_raises(CCLError, ccl.sigmaM, cosmo, mhalo_arr, a)

    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_scl, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_lst, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_arr, a_arr)

def check_massfunc_MG(cosmo):
    """
    Check that mass function and supporting functions can be run.
    """

    z = 0.
    z_arr = np.linspace(0., 2., 10)
    a = 1.
    a_arr = 1. / (1.+z_arr)
    mhalo_scl = 1e13
    mhalo_lst = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    mhalo_arr = np.array([1e11, 1e12, 1e13, 1e14, 1e15, 1e16])
    odelta = 200.
    
    # massfunc
    assert_raises( RuntimeError, ccl.massfunc, cosmo, mhalo_scl, a, odelta)
    assert_raises( RuntimeError, ccl.massfunc, cosmo, mhalo_lst, a, odelta)
    assert_raises( RuntimeError, ccl.massfunc, cosmo, mhalo_arr, a, odelta)
    
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_scl, a_arr, odelta)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_lst, a_arr, odelta)
    assert_raises(TypeError, ccl.massfunc, cosmo, mhalo_arr, a_arr, odelta)
    
    # Check whether odelta out of bounds
    assert_raises(RuntimeError, ccl.massfunc, cosmo, mhalo_scl, a, 199.)
    assert_raises(RuntimeError, ccl.massfunc, cosmo, mhalo_scl, a, 5000.)
    
    # massfunc_m2r
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_scl)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_lst)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_arr)) )
    
    # sigmaM
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_scl, a)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_lst, a)) )
    assert_( all_finite(ccl.sigmaM(cosmo, mhalo_arr, a)) )
    
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_scl, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_lst, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_arr, a_arr)
    
    # halo_bias
    assert_raises( RuntimeError, ccl.halo_bias, cosmo, mhalo_scl, a)
    assert_raises( RuntimeError, ccl.halo_bias, cosmo, mhalo_lst, a)
    assert_raises( RuntimeError, ccl.halo_bias, cosmo, mhalo_arr, a)

def check_halomod(cosmo):
    """
    Check that halo model functions can be run.
    """

    # Time variables
    z = 0.
    z_array = np.linspace(0., 2., 10)
    a = 1.
    a_array = 1. / (1.+z_array)

    # Halo definition
    odelta = 200.

    # Mass variables
    mass_scalar = 1e13
    mass_list = [1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
    mass_array = np.array([1e11, 1e12, 1e13, 1e14, 1e15, 1e16])

    # Wave-vector variables
    k_scalar = 1.
    k_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    k_array = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])

    # halo concentration
    assert_( all_finite(ccl.halo_concentration(cosmo, mass_scalar, a, odelta)) )
    assert_( all_finite(ccl.halo_concentration(cosmo, mass_list,   a, odelta)) )
    assert_( all_finite(ccl.halo_concentration(cosmo, mass_array,  a, odelta)) )

    assert_raises(TypeError, ccl.halo_concentration, cosmo, mass_scalar, a_array, odelta)
    assert_raises(TypeError, ccl.halo_concentration, cosmo, mass_list,   a_array, odelta)
    assert_raises(TypeError, ccl.halo_concentration, cosmo, mass_array,  a_array, odelta)

    # halo model
    assert_( all_finite(ccl.halomodel_matter_power(cosmo, k_scalar, a)) )
    assert_( all_finite(ccl.halomodel_matter_power(cosmo, k_list,   a)) )
    assert_( all_finite(ccl.halomodel_matter_power(cosmo, k_array,  a)) )

    assert_( all_finite(ccl.halomodel.twohalo_matter_power(cosmo, k_scalar, a)) )
    assert_( all_finite(ccl.halomodel.twohalo_matter_power(cosmo, k_list,   a)) )
    assert_( all_finite(ccl.halomodel.twohalo_matter_power(cosmo, k_array,  a)) )

    assert_( all_finite(ccl.halomodel.onehalo_matter_power(cosmo, k_scalar, a)) )
    assert_( all_finite(ccl.halomodel.onehalo_matter_power(cosmo, k_list,   a)) )
    assert_( all_finite(ccl.halomodel.onehalo_matter_power(cosmo, k_array,  a)) )

    assert_raises(TypeError, ccl.halomodel_matter_power, cosmo, k_scalar, a_array)
    assert_raises(TypeError, ccl.halomodel_matter_power, cosmo, k_list,   a_array)
    assert_raises(TypeError, ccl.halomodel_matter_power, cosmo, k_array,  a_array)

def check_neutrinos():
    """
    Check that neutrino-related functions can be run.
    """
    z = 0.
    z_arr = np.linspace(0., 2., 10)
    a = 1.
    a_arr = 1. / (1.+z_arr)
    a_lst = [_a for _a in a_arr]

    T_CMB = 2.725
    N_nu_mass = 3
    mnu = [0.02, 0.02, 0.02]

    # Omeganuh2
    assert_( all_finite(ccl.Omeganuh2(a, mnu, T_CMB)) )
    assert_( all_finite(ccl.Omeganuh2(a_lst, mnu, T_CMB)) )
    assert_( all_finite(ccl.Omeganuh2(a_arr, mnu, T_CMB)) )

    OmNuh2 = 0.01

    # Omeganuh2_to_Mnu
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'normal', T_CMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'inverted', T_CMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'equal', T_CMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'sum', T_CMB)) )

    # Check that the right exceptions are raised
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.27, Omega_b=0.045,
                                             h=0.67, A_s=1e-10, n_s=0.96,
                                             m_nu=[0.1, 0.2, 0.3, 0.4])
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.27, Omega_b=0.045,
                                             h=0.67, A_s=1e-10, n_s=0.96,
                                             m_nu=[0.1, 0.2, 0.3],
                                             mnu_type="sum")
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.27, Omega_b=0.045,
                                             h=0.67, A_s=1e-10, n_s=0.96,
                                             m_nu=42)


def check_cls(cosmo):
    """
    Check that cls functions can be run.
    """
    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)

    # Bias input
    b = np.sqrt(1. + z)

    # ell range input
    ell_scl = 4
    ell_lst = [2, 3, 4, 5, 6, 7, 8, 9]
    ell_arr = np.arange(2, 10)

    # Check if power spectrum type is valid for CMB
    cmb_ok = True
    if cosmo._config.matter_power_spectrum_method \
        == ccl.core.matter_power_spectrum_types['emu']: cmb_ok = False

    # ClTracer test objects
    lens1 = ccl.WeakLensingTracer(cosmo, (z, n))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z,n), ia_bias=(z, n), red_frac=(z,n))
    nc1 = ccl.NumberCountsTracer(cosmo, False, dndz=(z,n), bias=(z,b))
    nc2 = ccl.NumberCountsTracer(cosmo, True, dndz=(z,n), bias=(z,b))
    nc3 = ccl.NumberCountsTracer(cosmo, True, dndz=(z,n), bias=(z,b), mag_bias=(z,b))
    cmbl=ccl.CMBLensingTracer(cosmo, 1100.)

    assert_raises(ValueError, ccl.WeakLensingTracer, cosmo, None)

    # Check valid ell input is accepted
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_arr)) )

    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr)) )

    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, cmbl, cmbl, ell_arr)) )

    # Check non-limber calculations
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr, l_limber=20)))
    # Non-Limber only implemented for number counts
    assert_raises(CCLError, ccl.angular_cl, cosmo, lens1, lens1, ell_arr, l_limber=20)

    # Check various cross-correlation combinations
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc3, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, lens1, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc3, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, lens2, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc3, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, nc1, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc2, nc3, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, nc2, cmbl, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, nc3, cmbl, ell_arr)) )

    # Check that reversing order of ClTracer inputs works
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens2, ell_arr)) )



def check_cls_nu(cosmo):
    """
    Check that cls functions can be run.
    """
    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)

    # Bias input
    b = np.sqrt(1. + z)

    # ell range input
    ell_scl = 4
    ell_lst = [2, 3, 4, 5, 6, 7, 8, 9]
    ell_arr = np.arange(2, 10)

    # Check if power spectrum type is valid for CMB
    cmb_ok = True
    if cosmo._config.matter_power_spectrum_method \
        == ccl.core.matter_power_spectrum_types['emu']: cmb_ok = False

    # ClTracer test objects
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z,n))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z,n), ia_bias=(z,n), red_frac=(z,n))
    nc1 = ccl.NumberCountsTracer(cosmo, False, dndz=(z,n), bias=(z,b))

    # Check that for massive neutrinos including rsd raises an error (not yet implemented)
    assert_raises(CCLError, ccl.NumberCountsTracer, cosmo, True, dndz=(z,n), bias=(z,b))

    cmbl=ccl.CMBLensingTracer(cosmo,1100.)

    # Check valid ell input is accepted
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_arr)) )

    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr)) )

    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, cmbl, cmbl, ell_arr)) )

    # Check various cross-correlation combinations
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc1, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, lens1, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc1, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, lens2, cmbl, ell_arr)) )
    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, nc1, cmbl, ell_arr)) )

    # Check that reversing order of ClTracer inputs works
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens2, ell_arr)) )

    # Check get_internal_function()
    a_scl = 0.5
    a_lst = [0.2, 0.4, 0.6, 0.8, 1.]
    a_arr = np.linspace(0.2, 1., 5)
    assert_( all_finite(nc1.get_internal_function(cosmo, 'dndz', a_scl)) )
    assert_( all_finite(nc1.get_internal_function(cosmo, 'dndz', a_lst)) )
    assert_( all_finite(nc1.get_internal_function(cosmo, 'dndz', a_arr)) )

    # Check that invalid options raise errors
    assert_raises(ValueError, nc1.get_internal_function, cosmo, 'x', a_arr)
    assert_raises(CCLError, ccl.NumberCountsTracer, cosmo, True,
                  dndz=(z,n), bias=(z,b))
    assert_raises(ValueError, ccl.WeakLensingTracer, cosmo,
                  dndz=(z,n), ia_bias=(z,n))


def check_corr(cosmo):

    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.ones(z.shape)

    # ClTracer test objects
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z, n))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z,n), ia_bias=(z,n), red_frac=(z,n))

    ells = np.arange(3000)
    cls = ccl.angular_cl(cosmo, lens1, lens2, ells)

    t_arr = np.logspace(-2., np.log10(5.), 20) # degrees
    t_lst = [t for t in t_arr]
    t_scl = 2.
    t_int = 2

    # Make sure correlation functions work for valid inputs
    corr1 = ccl.correlation(cosmo, ells, cls, t_arr, corr_type='L+',
                            method='FFTLog')
    corr2 = ccl.correlation(cosmo, ells, cls, t_lst, corr_type='L+',
                            method='FFTLog')
    corr3 = ccl.correlation(cosmo, ells, cls, t_scl, corr_type='L+',
                            method='FFTLog')
    corr4 = ccl.correlation(cosmo, ells, cls, t_int, corr_type='L+',
                            method='FFTLog')
    assert_( all_finite(corr1))
    assert_( all_finite(corr2))
    assert_( all_finite(corr3))
    assert_( all_finite(corr4))

    # Check that exceptions are raised for invalid input
    assert_raises(ValueError, ccl.correlation, cosmo, ells, cls, t_arr,
                  corr_type='xx', method='FFTLog')
    assert_raises(ValueError, ccl.correlation, cosmo, ells, cls, t_arr,
                  corr_type='L+', method='xx')


def check_corr_3d(cosmo):

    # Scale factor
    a = 0.8

    # Distances (in Mpc)
    r_int = 50
    r = 50.
    r_lst = np.linspace(50,100,10)

    # Make sure correlation functions work for valid inputs
    corr1 = ccl.correlation_3d(cosmo, a, r_int)
    corr2 = ccl.correlation_3d(cosmo, a, r)
    corr3 = ccl.correlation_3d(cosmo, a, r_lst)
    assert_( all_finite(corr1))
    assert_( all_finite(corr2))
    assert_( all_finite(corr3))

def check_corr_3dRSD(cosmo):

    # Scale factor
    a = 0.8

    # Cosine of the angle
    mu = 0.7

    # Growth rate divided by galaxy bias
    beta = 0.5

    # Distances (in Mpc)
    s_int = 50
    s = 50.
    s_lst = np.linspace(50,100,10)

    # Make sure 3d correlation functions work for valid inputs
    corr1 = ccl.correlation_3dRsd(cosmo, a, s_int, mu, beta)
    corr2 = ccl.correlation_3dRsd(cosmo, a, s, mu, beta)
    corr3 = ccl.correlation_3dRsd(cosmo, a, s_lst, mu, beta)
    assert_( all_finite(corr1))
    assert_( all_finite(corr2))
    assert_( all_finite(corr3))

    corr4 = ccl.correlation_3dRsd_avgmu(cosmo, a, s_int, beta)
    corr5 = ccl.correlation_3dRsd_avgmu(cosmo, a, s, beta)
    corr6 = ccl.correlation_3dRsd_avgmu(cosmo, a, s_lst, beta)
    assert_( all_finite(corr4))
    assert_( all_finite(corr5))
    assert_( all_finite(corr6))

    corr7 = ccl.correlation_multipole(cosmo, a, beta, 0, s_lst)
    corr8 = ccl.correlation_multipole(cosmo, a, beta, 2, s_lst)
    corr9 = ccl.correlation_multipole(cosmo, a, beta, 4, s_lst)
    assert_( all_finite(corr7))
    assert_( all_finite(corr8))
    assert_( all_finite(corr9))

    # Distances (in Mpc)
    pie = 50.
    sig_int = 50
    sig = 50.
    sig_lst = np.linspace(50,100,10)

    corr10 = ccl.correlation_pi_sigma(cosmo, a, beta, pie, sig_int)
    corr11 = ccl.correlation_pi_sigma(cosmo, a, beta, pie, sig)
    corr12 = ccl.correlation_pi_sigma(cosmo, a, beta, pie, sig_lst)
    assert_( all_finite(corr10))
    assert_( all_finite(corr11))
    assert_( all_finite(corr12))

def test_background():
	
    """
    Test background and growth functions in ccl.background.
    """

    for cosmo in reference_models():
        yield check_background, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_background_nu, cosmo_nu
        
    for cosmo_mg in reference_models_mg():
		yield check_background, cosmo_mg

def test_power():
    
    """
    Test power spectrum and sigma functions in ccl.power.
    """
    
    for cosmo in reference_models():
        yield check_power, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_power, cosmo_nu
        
    for cosmo_mg in reference_models_mg():
		yield check_power_MG, cosmo_mg

@decorators.slow
def test_massfunc():
    
    """
    Test mass function and supporting functions.
    """
    
    for cosmo in reference_models():
        yield check_massfunc, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_massfunc_nu, cosmo_nu

def test_halomod():
    """
    Test halo model and supporting functions.
    """
    
    for cosmo in reference_models():
        yield check_halomod, cosmo

    for cosmo_mg in reference_models_mg():
        yield check_massfunc_MG, cosmo_mg

@decorators.slow
def test_neutrinos():
    """ 
    Test neutrino-related functions.
    """
    
    yield check_neutrinos

@decorators.slow
def test_cls():
    
    """
    Test top-level functions in pyccl.cls module.
    """
    
    for cosmo in reference_models():
        yield check_cls, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_cls_nu, cosmo_nu
        
    for cosmo_mg in reference_models_mg():
		yield check_cls, cosmo_mg
	
def test_corr():

    """
    Test top-level functions in pyccl.correlation module.
    """

    for cosmo in reference_models():
        yield check_corr, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_corr, cosmo_nu

    for cosmo in reference_models():
        yield check_corr_3d, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_corr_3d, cosmo_nu
        
    for cosmo_mg in reference_models_mg():
		yield check_corr, cosmo_mg
		
    for cosmo_mg in reference_models_mg():
		yield check_corr_3d, cosmo_mg

    for cosmo in reference_models():
        yield check_corr_3dRSD, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_corr_3dRSD, cosmo_nu

def test_debug_mode():

    """
    Test that debug mode can be toggled.
    """

    ccl.debug_mode(True)
    ccl.debug_mode(False)


if __name__ == '__main__':
    run_module_suite()
