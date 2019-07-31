import numpy as np
import pyccl as ccl
import pytest


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
    assert np.allclose(cosmo['mnu'], 0)
    assert np.allclose(cosmo['Omega_n_mass'], 0)
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
    #     Omega_n_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_n_rel'] + cosmo['Omega_n_mass'] + cosmo['Omega_k'],
        1)


@pytest.mark.parametrize('mnu_type', ['sum', 'sum_inverted'])
def test_parametes_nu(mnu_type):
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
        mnu_type=mnu_type
    )

    assert np.allclose(
        cosmo['mnu'][1]**2 - cosmo['mnu'][0]**2,
        ccl.physical_constants.DELTAM12_sq, atol=1e-4, rtol=0)

    if mnu_type == 'sum_inverted':
        assert np.allclose(
            cosmo['mnu'][2]**2 - cosmo['mnu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_neg, atol=1e-4, rtol=0)
        pass
    else:
        assert np.allclose(
            cosmo['mnu'][2]**2 - cosmo['mnu'][0]**2,
            ccl.physical_constants.DELTAM13_sq_pos, atol=1e-4, rtol=0)


def test_parameters_nu_list():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=[0.1, 0.01, 0.003],
        mnu_type='list')

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
    #     Omega_n_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_n_rel'] + cosmo['Omega_n_mass'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.1 + 0.01 + 0.003)
    assert np.allclose(cosmo['mnu'], [0.1, 0.01, 0.003])


def test_parameters_nu_sum():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        mnu_type='sum')

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
    #     Omega_n_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_n_rel'] + cosmo['Omega_n_mass'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)


def test_parameters_nu_sum_inverted():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        mnu_type='sum_inverted')

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
    #     Omega_n_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_n_rel'] + cosmo['Omega_n_mass'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)


def test_parameters_nu_sum_equal():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        m_nu=0.3,
        mnu_type='sum_equal')

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
    #     Omega_n_rel
    #     Omega_g
    #     Omega_l
    assert np.allclose(
        cosmo['Omega_l'] + cosmo['Omega_m'] + cosmo['Omega_g'] +
        cosmo['Omega_n_rel'] + cosmo['Omega_n_mass'] + cosmo['Omega_k'],
        1)

    assert np.allclose(cosmo['Neff'], 3.046)
    assert cosmo['N_nu_mass'] == 3
    assert np.allclose(cosmo['sum_nu_masses'], 0.3)
    assert np.allclose(cosmo['mnu'], 0.1)


@pytest.mark.parametrize('m_nu,kind', [(0.05, 'sum'), (0.09, 'sum_inverted')])
def test_parameters_nu_unphysical_raises(m_nu, kind):
    with pytest.raises(ValueError):
        ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            A_s=2.1e-9,
            n_s=0.96,
            m_nu=m_nu,
            mnu_type=kind)
