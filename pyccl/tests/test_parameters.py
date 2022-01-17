import tempfile
import numpy as np
import pytest

import pyccl as ccl


def test_parameters_lcdm_defaults():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96)

    assert np.allclose(cosmo['Omega_c'], 0.25)
    assert np.allclose(cosmo['Omega_b'], 0.05)
    assert np.allclose(cosmo['Omega_m'], 0.30)
    assert np.allclose(cosmo['Omega_k'], 0)
    assert np.allclose(cosmo['sqrtk'], 0)
    assert np.allclose(cosmo['k_sign'], 0)
    assert np.allclose(cosmo['w0'], -1)
    assert np.allclose(cosmo['wa'], 0)
    assert np.allclose(cosmo['H0'], 70)
    assert np.allclose(cosmo['h'], 0.7)
    assert np.allclose(cosmo['A_s'], 2.1e-9)
    assert np.allclose(cosmo['n_s'], 0.96)
    assert np.isnan(cosmo['sigma8'])
    assert np.isnan(cosmo['z_star'])
    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 0
    assert np.allclose(cosmo['N_nu_rel'], 3.046)
    assert np.allclose(cosmo['sum_nu_masses'], 0)
    assert np.allclose(cosmo['m_nu'], 0)
    assert np.allclose(cosmo['Omega_nu_mass'], 0)
    assert np.allclose(cosmo['T_CMB'], ccl.physical_constants.T_CMB)

    assert np.allclose(cosmo['bcm_ks'], 55.0)
    assert np.allclose(cosmo['bcm_log10Mc'], np.log10(1.2e14))
    assert np.allclose(cosmo['bcm_etab'], 0.5)

    assert not cosmo['has_mgrowth']
    assert cosmo['nz_mgrowth'] == 0
    assert cosmo['z_mgrowth'] is None
    assert cosmo['df_mgrowth'] is None

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_nu_mass'] + cosmo['Omega_k'],
        1)


@pytest.mark.parametrize('m_nu_type', ['normal', 'inverted', 'single'])
def test_parameters_nu(m_nu_type):
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        wa=0.01,
        w0=-1,
        Neff=3.046,
        Omega_k=0.0,
        m_nu=0.15,
        m_nu_type=m_nu_type
    )

    if m_nu_type == 'inverted':
        assert np.allclose(cosmo['m_nu'][1]**2 - cosmo['m_nu'][0]**2,
                           ccl.physical_constants.DELTAM12_sq,
                           atol=1e-4, rtol=0)
        assert np.allclose(
            cosmo['m_nu'][2]**2 - cosmo['m_nu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_neg, atol=1e-4, rtol=0)
    elif m_nu_type == 'normal':
        assert np.allclose(cosmo['m_nu'][1]**2 - cosmo['m_nu'][0]**2,
                           ccl.physical_constants.DELTAM12_sq,
                           atol=1e-4, rtol=0)
        assert np.allclose(
            cosmo['m_nu'][2]**2 - cosmo['m_nu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_pos, atol=1e-4, rtol=0)
    elif m_nu_type == 'single':
        assert np.allclose(cosmo['m_nu'][0], 0.15, atol=1e-4, rtol=0)
        assert np.allclose(cosmo['m_nu'][1], 0., atol=1e-4, rtol=0)
        assert np.allclose(cosmo['m_nu'][2], 0., atol=1e-4, rtol=0)


def test_parameters_nu_Nnurel_neg():
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=0.67, sigma8=0.8,
                      n_s=0.96, m_nu=[0.03, 0.02, 0.04], Neff=3.,
                      m_nu_type='list')


def test_parameters_nu_list():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=[0.1, 0.01, 0.003],
        m_nu_type='list')

    assert np.allclose(cosmo['Omega_c'], 0.25)
    assert np.allclose(cosmo['Omega_b'], 0.05)
    assert np.allclose(cosmo['Omega_m'] - cosmo['Omega_nu_mass'], 0.30)
    assert np.allclose(cosmo['Omega_k'], 0)
    assert np.allclose(cosmo['sqrtk'], 0)
    assert np.allclose(cosmo['k_sign'], 0)
    assert np.allclose(cosmo['w0'], -1)
    assert np.allclose(cosmo['wa'], 0)
    assert np.allclose(cosmo['H0'], 70)
    assert np.allclose(cosmo['h'], 0.7)
    assert np.allclose(cosmo['A_s'], 2.1e-9)
    assert np.allclose(cosmo['n_s'], 0.96)
    assert np.isnan(cosmo['sigma8'])
    assert np.isnan(cosmo['z_star'])
    assert np.allclose(cosmo['T_CMB'], ccl.physical_constants.T_CMB)

    assert np.allclose(cosmo['bcm_ks'], 55.0)
    assert np.allclose(cosmo['bcm_log10Mc'], np.log10(1.2e14))
    assert np.allclose(cosmo['bcm_etab'], 0.5)

    assert not cosmo['has_mgrowth']
    assert cosmo['nz_mgrowth'] == 0
    assert cosmo['z_mgrowth'] is None
    assert cosmo['df_mgrowth'] is None

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.1 + 0.01 + 0.003)
    assert np.allclose(cosmo['m_nu'], [0.1, 0.01, 0.003])


def test_parameters_nu_normal():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        m_nu_type='normal')

    assert np.allclose(cosmo['Omega_c'], 0.25)
    assert np.allclose(cosmo['Omega_b'], 0.05)
    assert np.allclose(cosmo['Omega_m'] - cosmo['Omega_nu_mass'], 0.30)
    assert np.allclose(cosmo['Omega_k'], 0)
    assert np.allclose(cosmo['sqrtk'], 0)
    assert np.allclose(cosmo['k_sign'], 0)
    assert np.allclose(cosmo['w0'], -1)
    assert np.allclose(cosmo['wa'], 0)
    assert np.allclose(cosmo['H0'], 70)
    assert np.allclose(cosmo['h'], 0.7)
    assert np.allclose(cosmo['A_s'], 2.1e-9)
    assert np.allclose(cosmo['n_s'], 0.96)
    assert np.isnan(cosmo['sigma8'])
    assert np.isnan(cosmo['z_star'])
    assert np.allclose(cosmo['T_CMB'], ccl.physical_constants.T_CMB)

    assert np.allclose(cosmo['bcm_ks'], 55.0)
    assert np.allclose(cosmo['bcm_log10Mc'], np.log10(1.2e14))
    assert np.allclose(cosmo['bcm_etab'], 0.5)

    assert not cosmo['has_mgrowth']
    assert cosmo['nz_mgrowth'] == 0
    assert cosmo['z_mgrowth'] is None
    assert cosmo['df_mgrowth'] is None

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)


def test_parameters_nu_inverted():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        m_nu_type='inverted')

    assert np.allclose(cosmo['Omega_c'], 0.25)
    assert np.allclose(cosmo['Omega_b'], 0.05)
    assert np.allclose(cosmo['Omega_m'] - cosmo['Omega_nu_mass'], 0.30)
    assert np.allclose(cosmo['Omega_k'], 0)
    assert np.allclose(cosmo['sqrtk'], 0)
    assert np.allclose(cosmo['k_sign'], 0)
    assert np.allclose(cosmo['w0'], -1)
    assert np.allclose(cosmo['wa'], 0)
    assert np.allclose(cosmo['H0'], 70)
    assert np.allclose(cosmo['h'], 0.7)
    assert np.allclose(cosmo['A_s'], 2.1e-9)
    assert np.allclose(cosmo['n_s'], 0.96)
    assert np.isnan(cosmo['sigma8'])
    assert np.isnan(cosmo['z_star'])
    assert np.allclose(cosmo['T_CMB'], ccl.physical_constants.T_CMB)

    assert np.allclose(cosmo['bcm_ks'], 55.0)
    assert np.allclose(cosmo['bcm_log10Mc'], np.log10(1.2e14))
    assert np.allclose(cosmo['bcm_etab'], 0.5)

    assert not cosmo['has_mgrowth']
    assert cosmo['nz_mgrowth'] == 0
    assert cosmo['z_mgrowth'] is None
    assert cosmo['df_mgrowth'] is None

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)


def test_parameters_nu_equal():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        m_nu_type='equal')

    assert np.allclose(cosmo['Omega_c'], 0.25)
    assert np.allclose(cosmo['Omega_b'], 0.05)
    assert np.allclose(cosmo['Omega_m'] - cosmo['Omega_nu_mass'], 0.30)
    assert np.allclose(cosmo['Omega_k'], 0)
    assert np.allclose(cosmo['sqrtk'], 0)
    assert np.allclose(cosmo['k_sign'], 0)
    assert np.allclose(cosmo['w0'], -1)
    assert np.allclose(cosmo['wa'], 0)
    assert np.allclose(cosmo['H0'], 70)
    assert np.allclose(cosmo['h'], 0.7)
    assert np.allclose(cosmo['A_s'], 2.1e-9)
    assert np.allclose(cosmo['n_s'], 0.96)
    assert np.isnan(cosmo['sigma8'])
    assert np.isnan(cosmo['z_star'])
    assert np.allclose(cosmo['T_CMB'], ccl.physical_constants.T_CMB)

    assert np.allclose(cosmo['bcm_ks'], 55.0)
    assert np.allclose(cosmo['bcm_log10Mc'], np.log10(1.2e14))
    assert np.allclose(cosmo['bcm_etab'], 0.5)

    assert not cosmo['has_mgrowth']
    assert cosmo['nz_mgrowth'] == 0
    assert cosmo['z_mgrowth'] is None
    assert cosmo['df_mgrowth'] is None

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)
    assert np.allclose(cosmo['m_nu'], 0.1)


@pytest.mark.parametrize('m_nu,kind', [(0.05, 'normal'), (0.09, 'inverted')])
def test_parameters_nu_unphysical_raises(m_nu, kind):
    with pytest.raises(ValueError):
        ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            A_s=2.1e-9,
            n_s=0.96,
            m_nu=m_nu,
            m_nu_type=kind)


def test_parameters_valid_input():
    """
    Check that valid parameter arguments are accepted.
    """
    with pytest.warns(None) as w_rec:
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      Omega_k=0.05)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      Neff=2.046)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      Neff=3.046, m_nu=0.06)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      w0=-0.9)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      w0=-0.9, wa=0.1)
        # Check that kwarg order doesn't matter
        ccl.Cosmology(h=0.7, Omega_c=0.25, Omega_b=0.05, A_s=2.1e-9, n_s=0.96)
        # Try a set of parameters with non-zero mu0 / Sig0
        ccl.Cosmology(h=0.7, Omega_c=0.25, Omega_b=0.05, A_s=2.1e-9, n_s=0.96,
                      mu_0=0.1, sigma_0=0.1)
    assert len(w_rec) == 0


def test_parameters_missing():
    """
    Check that errors are raised when compulsory parameters are missing, but
    not when non-compulsory ones are.
    """

    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25)

    # Check that a single missing compulsory parameter is noticed
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, A_s=2.1e-9, n_s=0.96)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, h=0.7, A_s=2.1e-9, n_s=0.96)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96)

    # Make sure that compulsory parameters are compulsory
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, Omega_k=None)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, w0=None)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, wa=None)

    # Check that sigma8 vs A_s is handled ok.
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8,
                      A_s=2.1e-9, sigma8=0.7)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8)

    # Make sure that optional parameters are optional
    with pytest.warns(None) as w_rec:
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                      A_s=2.1e-9, n_s=0.96, z_mg=None, df_mg=None)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                      A_s=2.1e-9, n_s=0.96, z_mg=None)
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                      A_s=2.1e-9, n_s=0.96, df_mg=None)
    assert len(w_rec) == 0


def test_parameters_set():
    """
    Check that a Cosmology object doesn't let parameters be set.
    """
    params = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
        n_s=0.96)

    # Check that values of sigma8 and A_s won't be misinterpreted by the C code
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2e-5, n_s=0.96)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=9e-6,
                      n_s=0.96)

    # Check that error is raised when unrecognized parameter requested
    with pytest.raises(KeyError):
        (lambda: params['wibble'])()


def test_parameters_mgrowth():
    """
    Check that valid modified growth inputs are allowed, and invalid ones are
    rejected.
    """
    zarr = np.linspace(0., 1., 15)
    dfarr = 0.1 * np.ones(15)

    def f_func(z):
        return 0.1 * z

    # Valid constructions
    for omega_g in [None, 0.0, 0.1]:
        with pytest.warns(None) as w_rec:
            with pytest.warns(ccl.CCLDeprecationWarning):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96, z_mg=zarr,
                              df_mg=dfarr, Omega_g=omega_g)
        assert len(w_rec) == 0

        with pytest.warns(None) as w_rec:
            with pytest.warns(ccl.CCLDeprecationWarning):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96, z_mg=[0., 0.1, 0.2],
                              df_mg=[0.1, 0.1, 0.1], Omega_g=omega_g)
        assert len(w_rec) == 0

        # Invalid constructions
        with pytest.warns(ccl.CCLDeprecationWarning):
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=zarr, Omega_g=omega_g)
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              df_mg=dfarr, Omega_g=omega_g)
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=None,
                              df_mg=dfarr, Omega_g=omega_g)
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=zarr,
                              df_mg=0.1, Omega_g=omega_g)
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=zarr,
                              df_mg=f_func, Omega_g=omega_g)

            # Mis-matched array sizes and dimensionality
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=zarr,
                              df_mg=dfarr[1:], Omega_g=omega_g)
            with pytest.raises(ValueError):
                ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              A_s=2.1e-9, n_s=0.96,
                              z_mg=zarr,
                              df_mg=np.column_stack((dfarr, dfarr)),
                              Omega_g=omega_g)


def test_parameters_read_write():
    """Check that Cosmology objects can be read and written"""
    with pytest.warns(ccl.CCLDeprecationWarning):
        params = ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=[0.02, 0.1, 0.05], m_nu_type='list',
            z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

    # Make a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_file_name = tmpfile.name

    # Write out and then read in the parameters from that file
    with pytest.raises(IOError):
        params.write_yaml("/bogus/file/name")
    params.write_yaml(temp_file_name)
    with pytest.warns(ccl.CCLDeprecationWarning):
        params2 = ccl.Cosmology.read_yaml(temp_file_name)

    # Check the read-in params are equal to the written out ones
    assert np.allclose(params['Omega_c'], params2['Omega_c'])
    assert np.allclose(params['Neff'], params2['Neff'])
    assert np.allclose(params['sum_nu_masses'], params2['sum_nu_masses'])

    # check overriding parameters with kwargs
    with pytest.warns(ccl.CCLDeprecationWarning):
        params3 = ccl.Cosmology.read_yaml(temp_file_name,
                                          matter_power_spectrum='emu',
                                          n_s=1.1)
    # check unmodified parameters are the same
    assert np.allclose(params['Omega_c'], params3['Omega_c'])
    assert np.allclose(params['Neff'], params3['Neff'])
    # check new parameters and config correctly updated
    assert np.allclose(1.1, params3['n_s'])
    assert np.allclose(params['sum_nu_masses'], params3['sum_nu_masses'])
    assert params3._config_init_kwargs['matter_power_spectrum'] == 'emu'

    # Now make a file that will be deleted so it does not exist
    # and check the right error is raise
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        temp_file_name = tmpfile.name

    with pytest.raises(IOError):
        ccl.Cosmology.read_yaml(filename=temp_file_name)
    with pytest.raises(IOError):
        params.read_yaml(
            filename=temp_file_name+"/nonexistent_directory/params.yml")


def test_omega_k():
    """ Check that the value of Omega_k is within reasonable bounds. """
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                      n_s=0.96, Omega_k=-2)
