import numpy as np
import pytest
from numpy.testing import assert_raises
import pyccl as ccl


NCHI = 100
CHIMIN = 100.
CHIMAX = 1000.
COSMO = ccl.CosmologyVanillaLCDM()


def tkkaf(k1, k2, a, alpha=1., beta=1.):
    return 1./(k1**alpha*k2**beta)


def get_tk3d(alpha=1, beta=1):
    a_arr = np.linspace(0.1, 1., 10)
    k_arr = np.geomspace(1E-4, 1E3, 10)
    tkka_arr = np.array([tkkaf(k_arr[None, :],
                               k_arr[:, None], a,
                               alpha, beta)
                         for a in a_arr])
    return ccl.Tk3D(a_arr, np.log(k_arr),
                    tkk_arr=np.log(tkka_arr),
                    is_logt=True)


def pred_covar(l1, l2, alpha=1., beta=1.):
    lfac = 1/((l1+0.5)**alpha*(l2+0.5)**beta)
    xpn = alpha+beta-5
    chifac = (CHIMAX**xpn-CHIMIN**xpn)/xpn
    return lfac*chifac/(4*np.pi)


def get_tracer():
    chis = np.linspace(CHIMIN, CHIMAX, NCHI)
    ws = np.ones(NCHI)
    t = ccl.Tracer()
    t.add_tracer(COSMO, kernel=(chis, ws))
    return t


@pytest.mark.parametrize("alpha,beta", [(3., 3.),
                                        (2., 2.),
                                        (1., 2.),
                                        (2., 1.)])
def test_cov_cNG_sanity(alpha, beta):
    # Compares covariance against analytical prediction
    tsp = get_tk3d(alpha, beta)
    tr = get_tracer()
    ls = np.array([2., 20., 200.])

    cov_p = pred_covar(ls[None, :], ls[:, None], alpha, beta)

    cov = ccl.angular_cl_cov_cNG(COSMO, tr, tr, ls, tsp)
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Spline integration (fast but inaccurate)
    cov = ccl.angular_cl_cov_cNG(COSMO, tr, tr, ls, tsp,
                                 integration_method='spline')
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 4E-2)

    # Different tracers
    cov = ccl.angular_cl_cov_cNG(COSMO, tr, tr, ls, tsp,
                                 cltracer3=tr, cltracer4=tr)
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Different ells
    cov = np.array([ccl.angular_cl_cov_cNG(COSMO, tr, tr, ls, tsp,
                                           ell2=l)
                    for l in ls])
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Different ells (transpose)
    cov = np.array([ccl.angular_cl_cov_cNG(COSMO, tr, tr, l, tsp,
                                           ell2=ls)
                    for l in ls]).T
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # All scalars
    cov = np.array([[ccl.angular_cl_cov_cNG(COSMO, tr, tr, l1, tsp,
                                            ell2=l2)
                     for l1 in ls]
                    for l2 in ls])
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)


def test_cov_cNG_errors():
    tsp = get_tk3d(1, 1)
    tr = get_tracer()
    ls = np.array([2., 20., 200.])

    assert_raises(ValueError, ccl.angular_cl_cov_cNG,
                  COSMO, tr, tr, ls, tsp,
                  integration_method='cag_cuad')

    assert_raises(ValueError, ccl.angular_cl_cov_cNG,
                  COSMO, tr, tr, ls, tr)
