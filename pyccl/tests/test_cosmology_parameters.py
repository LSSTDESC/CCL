import tempfile
import numpy as np
import pytest
import pyccl as ccl


def test_parameters_lcdmDefaultParams():
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
    assert np.allclose(cosmo['Neff'], 3.044)
    assert cosmo['N_nu_mass'] == 0
    assert np.allclose(cosmo['N_nu_rel'], 3.044)
    assert np.allclose(cosmo['sum_nu_masses'], 0)
    assert np.allclose(cosmo['m_nu'], 0)
    assert np.allclose(cosmo['Omega_nu_mass'], 0)
    assert np.allclose(cosmo['T_CMB'], ccl.cosmology.DefaultParams.T_CMB)

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_nu_mass'] + cosmo['Omega_k'],
        1)


@pytest.mark.parametrize('mass_split', ['normal', 'inverted', 'single'])
def test_parameters_nu(mass_split):
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        wa=0.01,
        w0=-1,
        Neff=3.044,
        Omega_k=0.0,
        m_nu=0.15,
        mass_split=mass_split
    )

    if mass_split == 'inverted':
        assert np.allclose(cosmo['m_nu'][1]**2 - cosmo['m_nu'][0]**2,
                           ccl.physical_constants.DELTAM12_sq,
                           atol=1e-4, rtol=0)
        assert np.allclose(
            cosmo['m_nu'][2]**2 - cosmo['m_nu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_neg, atol=1e-4, rtol=0)
    elif mass_split == 'normal':
        assert np.allclose(cosmo['m_nu'][1]**2 - cosmo['m_nu'][0]**2,
                           ccl.physical_constants.DELTAM12_sq,
                           atol=1e-4, rtol=0)
        assert np.allclose(
            cosmo['m_nu'][2]**2 - cosmo['m_nu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_pos, atol=1e-4, rtol=0)
    elif mass_split == 'single':
        assert len(cosmo["m_nu"]) == 1
        assert np.allclose(cosmo['m_nu'][0], 0.15, atol=1e-4, rtol=0)


def test_parameters_nu_Nnurel_neg():
    with pytest.raises(ValueError):
        ccl.Cosmology(
            Omega_c=0.27, Omega_b=0.049,
            h=0.67, sigma8=0.8, n_s=0.96, m_nu=[0.03, 0.02, 0.04],
            Neff=3., mass_split='list')


def test_parameters_nu_list():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=[0.1, 0.01, 0.003],
        mass_split='list')

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
    assert np.allclose(cosmo['T_CMB'], ccl.cosmology.DefaultParams.T_CMB)

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.044)
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
        mass_split='normal')

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
    assert np.allclose(cosmo['T_CMB'], ccl.cosmology.DefaultParams.T_CMB)

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.044)
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
        mass_split='inverted')

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
    assert np.allclose(cosmo['T_CMB'], ccl.cosmology.DefaultParams.T_CMB)

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.044)
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
        mass_split='equal')

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
    assert np.allclose(cosmo['T_CMB'], ccl.cosmology.DefaultParams.T_CMB)

    # these are defined in the code via some constants so
    # going to test the total
    #     Omega_nu_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_nu_rel'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.044)
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
            mass_split=kind)


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

    # Check that sigma8 vs A_s is handled ok.
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8,
                      A_s=2.1e-9, sigma8=0.7)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8)


def test_parameters_read_write():
    """Check that Cosmology objects can be read and written"""
    params = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], mass_split='list')

    # Make a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_file_name = tmpfile.name

    # Write out and then read in the parameters from that file
    with pytest.raises(IOError):
        params.write_yaml("/bogus/file/name")
    params.write_yaml(temp_file_name)
    params2 = ccl.Cosmology.read_yaml(temp_file_name)

    # Check the read-in params are equal to the written out ones
    assert params["Omega_c"] == params2["Omega_c"]
    assert params["Neff"] == params2["Neff"]
    assert params["sum_nu_masses"] == params2["sum_nu_masses"]

    # check overriding parameters with kwargs
    params3 = ccl.Cosmology.read_yaml(temp_file_name,
                                      matter_power_spectrum='linear',
                                      n_s=1.1)
    # check unmodified parameters are the same
    assert params["Omega_c"] == params3["Omega_c"]
    assert params["Neff"] == params3["Neff"]
    # check new parameters and config correctly updated
    assert params3["n_s"] == 1.1
    assert params["sum_nu_masses"] == params3["sum_nu_masses"]
    assert params3.matter_power_spectrum_type == 'linear'

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
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                      A_s=2.1e-9, n_s=0.96, Omega_k=-2)
