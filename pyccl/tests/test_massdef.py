import numpy as np
import pyccl as ccl


def test_mdef_eq():
    hmd_200m = ccl.halos.MassDef200mat()
    hmd_200m_b = ccl.halos.MassDef(200, 'matter')
    assert hmd_200m == hmd_200m_b


def test_concentration_translation():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                          h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                          w0=-1, wa=0, T_CMB=2.7)

    c_old = np.array([9., 10., 11.])
    Delta_old = 200.

    # No change expected
    Delta_new = 200.
    c_new = ccl.massdef.convert_concentration_py(cosmo,
                                                 c_old, Delta_old,
                                                 Delta_new)
    assert np.all(c_old == c_new)

    # Test against numerical solutions from Mathematica.
    Delta_new = 500.
    c_new = ccl.massdef.convert_concentration_py(cosmo,
                                                 c_old, Delta_old,
                                                 Delta_new)
    c_new_expected = np.array([6.12194, 6.82951, 7.53797])
    assert np.all(np.fabs(c_new/c_new_expected-1) < 1E-4)
