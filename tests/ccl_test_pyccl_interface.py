import numpy as np,math
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, decorators, run_module_suite
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

    # BBKS Pk
    p4 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    cosmo4 = ccl.Cosmology(p4,transfer_function='bbks')

    # E&H Pk
    p5 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    cosmo5 = ccl.Cosmology(p5,transfer_function='eisenstein_hu')

    # Baryons Pk
    p6 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    cosmo6 = ccl.Cosmology(p6,baryons_power_spectrum='bcm')
    
    # Baryons Pk with choice of BCM parameters other than default
    p7 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96,
                        bcm_log10Mc=math.log10(1.7e14), bcm_etab=0.3, bcm_ks=75.)
    cosmo7 = ccl.Cosmology(p7,baryons_power_spectrum='bcm')

    # Return 
    return [cosmo1,cosmo4,cosmo5,cosmo7] # cosmo2, cosmo3, cosmo6

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
    
    # omega_m_a
    assert_( all_finite(ccl.omega_x(cosmo, a_scl, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_lst, 'matter')) )
    assert_( all_finite(ccl.omega_x(cosmo, a_arr, 'matter')) )


def check_power(cosmo):
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


def check_cls(cosmo):
    """
    Check that cls functions can be run.
    """
    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.ones(z.shape)
    
    # Bias input
    b = np.sqrt(1. + z)
    
    # ell range input
    ell_scl = 4
    ell_lst = [2, 3, 4, 5, 6, 7, 8, 9]
    ell_arr = np.arange(2, 10)
    
    # ClTracer test objects
    lens1 = ccl.ClTracerLensing(cosmo, False, n=n, z=z)
    lens2 = ccl.ClTracerLensing(cosmo, True, n=(z,n), bias_ia=(z,n), f_red=(z,n))
    nc1 = ccl.ClTracerNumberCounts(cosmo, False, False, n=(z,n), bias=(z,b))
    nc2 = ccl.ClTracerNumberCounts(cosmo, True, False, n=(z,n), bias=(z,b))
    nc3 = ccl.ClTracerNumberCounts(cosmo, True, True, n=(z,n), bias=(z,b),
                                   mag_bias=(z,b))
    cmbl=ccl.ClTracerCMBLensing(cosmo,1100.)
    
    # Check valid ell input is accepted
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens1, ell_arr)) )
    
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_scl)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_lst)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr)) )

    assert_( all_finite(ccl.angular_cl(cosmo, cmbl, cmbl, ell_arr)) )
    
    # Check various cross-correlation combinations
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, lens2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, nc3, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens1, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, nc3, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, lens2, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc2, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc3, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc2, nc3, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc2, cmbl, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc3, cmbl, ell_arr)) )
    
    # Check that reversing order of ClTracer inputs works
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens1, ell_arr)) )
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, lens2, ell_arr)) )
    
def check_corr(cosmo):
    
    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.ones(z.shape)

    # ClTracer test objects
    lens1 = ccl.ClTracerLensing(cosmo, False, n=n, z=z)
    lens2 = ccl.ClTracerLensing(cosmo, True, n=(z,n), bias_ia=(z,n), f_red=(z,n))

    ells=np.arange(3000)
    cls=ccl.angular_cl(cosmo,lens1,lens2,ells)

    t=np.logspace(-2,np.log10(5.),20) #degrees
    corrfunc=ccl.correlation(cosmo,ells,cls,t,corr_type='L+',method='FFTLog')
    assert_( all_finite(corrfunc))
    
def test_background():
    """
    Test background and growth functions in ccl.background.
    """
    for cosmo in reference_models():
        yield check_background, cosmo

@decorators.slow
def test_power():
    """
    Test power spectrum and sigma functions in ccl.power.
    """
    for cosmo in reference_models():
        yield check_power, cosmo

@decorators.slow
def test_massfunc():
    """
    Test mass function and supporting functions.
    """
    for cosmo in reference_models():
        yield check_massfunc, cosmo

def test_lsst_specs():
    """
    Test lsst specs module.
    """
    for cosmo in reference_models():
        yield check_lsst_specs, cosmo

def test_cls():
    """
    Test top-level functions in pyccl.cls module.
    """
    for cosmo in reference_models():
        yield check_cls, cosmo

def test_corr():
    """
    Test top-level functions in pyccl.correlation module.
    """
    for cosmo in reference_models():
        yield check_corr, cosmo

        
if __name__ == '__main__':
    run_module_suite()
