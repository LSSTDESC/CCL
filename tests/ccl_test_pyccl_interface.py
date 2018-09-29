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
    cosmo4 = ccl.Cosmology(p4, transfer_function='bbks')

    # E&H Pk
    p5 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    cosmo5 = ccl.Cosmology(p5, transfer_function='eisenstein_hu')

    # Emulator Pk
    p6 = ccl.Parameters(Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
                        n_s=0.96, Neff=3.04, m_nu=0.)
    cosmo6 = ccl.Cosmology(p6, transfer_function='emulator',
                           matter_power_spectrum='emu')

    # Baryons Pk
    p8 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)
    cosmo8 = ccl.Cosmology(p8, baryons_power_spectrum='bcm')

    # Baryons Pk with choice of BCM parameters other than default
    p9 = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96,
                        bcm_log10Mc=math.log10(1.7e14), bcm_etab=0.3, bcm_ks=75.)
    cosmo9 = ccl.Cosmology(p9, baryons_power_spectrum='bcm')

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
    p1 = ccl.Parameters(Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
                        n_s=0.96, Neff=3.04, m_nu=[0.02, 0.02, 0.02])
    cosmo1 = ccl.Cosmology(p1, transfer_function='emulator',
                           matter_power_spectrum='emu')

    # Emulator Pk with neutrinos, force equalize
    #p2 = ccl.Parameters(Omega_c=0.27, Omega_b=0.022/0.67**2, h=0.67, sigma8=0.8,
    #                    n_s=0.96, Neff=3.04, m_nu=0.11)
    #cosmo2 = ccl.Cosmology(p1, transfer_function='emulator',
    #                       matter_power_spectrum='emu', emulator_neutrinos='equalize')

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
    assert_raises(RuntimeError, ccl.growth_factor, cosmo, a_scl)
    assert_raises(RuntimeError, ccl.growth_factor, cosmo, a_lst)
    assert_raises(RuntimeError, ccl.growth_factor, cosmo, a_arr)

    # growth_factor_unnorm
    assert_raises(RuntimeError, ccl.growth_factor_unnorm, cosmo, a_scl)
    assert_raises(RuntimeError, ccl.growth_factor_unnorm, cosmo, a_lst)
    assert_raises(RuntimeError, ccl.growth_factor_unnorm, cosmo, a_arr)

    # growth_rate
    assert_raises(RuntimeError,ccl.growth_rate, cosmo, a_scl)
    assert_raises(RuntimeError,ccl.growth_rate, cosmo, a_lst)
    assert_raises(RuntimeError,ccl.growth_rate, cosmo, a_arr)

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

    # sigmaV
    assert_( all_finite(ccl.sigmaV(cosmo, R_scl)) )
    assert_( all_finite(ccl.sigmaV(cosmo, R_lst)) )
    assert_( all_finite(ccl.sigmaV(cosmo, R_arr)) )

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
    assert_raises(RuntimeError, ccl.massfunc,cosmo, mhalo_scl, a, odelta)
    assert_raises(RuntimeError, ccl.massfunc,cosmo, mhalo_lst, a, odelta)
    assert_raises(RuntimeError, ccl.massfunc,cosmo, mhalo_arr, a, odelta)

    # halo bias
    assert_raises(RuntimeError, ccl.halo_bias, cosmo, mhalo_scl, a, odelta)
    assert_raises(RuntimeError, ccl.halo_bias, cosmo, mhalo_lst, a, odelta)
    assert_raises(RuntimeError, ccl.halo_bias, cosmo, mhalo_arr, a, odelta)

    # massfunc_m2r
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_scl)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_lst)) )
    assert_( all_finite(ccl.massfunc_m2r(cosmo, mhalo_arr)) )

    # sigmaM
    assert_raises(RuntimeError, ccl.sigmaM, cosmo, mhalo_scl, a)
    assert_raises(RuntimeError, ccl.sigmaM, cosmo, mhalo_lst, a)
    assert_raises(RuntimeError, ccl.sigmaM, cosmo, mhalo_arr, a)

    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_scl, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_lst, a_arr)
    assert_raises(TypeError, ccl.sigmaM, cosmo, mhalo_arr, a_arr)


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

    TCMB = 2.725
    N_nu_mass = 3
    mnu = [0.02, 0.02, 0.02]

    # Omeganuh2
    assert_( all_finite(ccl.Omeganuh2(a, mnu, TCMB)) )
    assert_( all_finite(ccl.Omeganuh2(a_lst, mnu, TCMB)) )
    assert_( all_finite(ccl.Omeganuh2(a_arr, mnu, TCMB)) )

    OmNuh2 = 0.01

    # Omeganuh2_to_Mnu
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'normal', TCMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'inverted', TCMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'equal', TCMB)) )
    assert_( all_finite(ccl.nu_masses(OmNuh2, 'sum', TCMB)) )

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
    PZ3 = ccl.PhotoZGaussian(sigma_z0=0.1)

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

def check_lsst_specs_nu(cosmo):
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
    assert_raises(RuntimeError,ccl.bias_clustering, cosmo, a_scl)
    assert_raises(RuntimeError,ccl.bias_clustering, cosmo, a_lst)
    assert_raises(RuntimeError,ccl.bias_clustering, cosmo, a_arr)

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
    n = np.exp(-((z-0.5)/0.1)**2)

    # Bias input
    b = np.sqrt(1. + z)

    # ell range input
    ell_scl = 4
    ell_lst = [2, 3, 4, 5, 6, 7, 8, 9]
    ell_arr = np.arange(2, 10)

    # Check if power spectrum type is valid for CMB
    cmb_ok = True
    if cosmo.configuration.matter_power_spectrum_method \
        == ccl.core.matter_power_spectrum_types['emu']: cmb_ok = False

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

    if cmb_ok: assert_( all_finite(ccl.angular_cl(cosmo, cmbl, cmbl, ell_arr)) )

    # Check non-limber calculations
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr, l_limber=20, non_limber_method="native")))
    assert_( all_finite(ccl.angular_cl(cosmo, nc1, nc1, ell_arr, l_limber=20, non_limber_method="angpow")))

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

    # Wrong non limber method
    assert_raises(ValueError, ccl.angular_cl, cosmo, lens1, lens1, ell_scl, non_limber_method='xx')



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
    if cosmo.configuration.matter_power_spectrum_method \
        == ccl.core.matter_power_spectrum_types['emu']: cmb_ok = False

    # ClTracer test objects
    lens1 = ccl.ClTracerLensing(cosmo, False, n=n, z=z)
    lens2 = ccl.ClTracerLensing(cosmo, True, n=(z,n), bias_ia=(z,n), f_red=(z,n))
    nc1 = ccl.ClTracerNumberCounts(cosmo, False, False, n=(z,n), bias=(z,b))

    # Check that for massive neutrinos including rsd raises an error (not yet implemented)
    assert_raises(RuntimeError, ccl.ClTracerNumberCounts, cosmo, True, False, n=(z,n), bias=(z,b))

    cmbl=ccl.ClTracerCMBLensing(cosmo,1100.)

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
    assert_raises(ValueError, ccl.ClTracerNumberCounts, cosmo, True, True,
                  n=(z,n), bias=(z,b))
    assert_raises(ValueError, ccl.ClTracer, cosmo, 'x', True, True,
                  n=(z,n), bias=(z,b))
    assert_raises(ValueError, ccl.ClTracerLensing, cosmo,
                  has_intrinsic_alignment=True, n=(z,n), bias_ia=(z,n))
    assert_no_warnings(ccl.cls._cltracer_obj, nc1)
    assert_no_warnings(ccl.cls._cltracer_obj, nc1.cltracer)
    assert_raises(TypeError, ccl.cls._cltracer_obj, None)


def check_corr(cosmo):

    # Number density input
    z = np.linspace(0., 1., 200)
    n = np.ones(z.shape)

    # ClTracer test objects
    lens1 = ccl.ClTracerLensing(cosmo, False, n=n, z=z)
    lens2 = ccl.ClTracerLensing(cosmo, True, n=(z,n), bias_ia=(z,n), f_red=(z,n))

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



def test_valid_transfer_combos():
    """
    Check that invalid transfer_function <-> matter_power_spectrum pairs raise
    an error.
    """
    params = { 'Omega_c': 0.27, 'Omega_b': 0.045, 'h': 0.67,
               'A_s': 1e-10, 'n_s': 0.96, 'w0': -1., 'wa': 0. }

    assert_raises(ValueError, ccl.Cosmology, transfer_function='emulator',
                              matter_power_spectrum='linear', **params)
    #assert_raises(ValueError, ccl.Cosmology, transfer_function='boltzmann',
    #                          matter_power_spectrum='halomodel', **params)
    assert_raises(ValueError, ccl.Cosmology, transfer_function='bbks',
                              matter_power_spectrum='emu', **params)

def test_background():
    """
    Test background and growth functions in ccl.background.
    """
    for cosmo in reference_models():
        yield check_background, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_background_nu, cosmo_nu

def test_power():
    """
    Test power spectrum and sigma functions in ccl.power.
    """
    for cosmo in reference_models():
        yield check_power, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_power, cosmo_nu

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

@decorators.slow
def test_neutrinos():
    """
    Test neutrino-related functions.
    """
    yield check_neutrinos

def test_lsst_specs():
    """
    Test lsst specs module.
    """
    for cosmo in reference_models():
        yield check_lsst_specs, cosmo

    for cosmo_nu in reference_models_nu():
       yield check_lsst_specs_nu, cosmo_nu

@decorators.slow
def test_cls():
    """
    Test top-level functions in pyccl.cls module.
    """
    for cosmo in reference_models():
        yield check_cls, cosmo

    for cosmo_nu in reference_models_nu():
        yield check_cls_nu, cosmo_nu

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

def test_debug_mode():
    """
    Test that debug mode can be toggled.
    """
    ccl.debug_mode(True)
    ccl.debug_mode(False)



if __name__ == '__main__':
    run_module_suite()
