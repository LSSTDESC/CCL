import numpy as np
import pytest
from numpy.testing import assert_raises
import pyccl as ccl


NCHI = 100
CHIMIN = 100.
CHIMAX = 1000.
COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks')


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


def pred_covar(l1, l2, alpha=1., beta=1.,
               prefac=1./(4*np.pi), chi_power=6):
    lfac = 1/((l1+0.5)**alpha*(l2+0.5)**beta)
    xpn = alpha+beta-chi_power+1
    chifac = (CHIMAX**xpn-CHIMIN**xpn)/xpn
    return lfac*chifac*prefac


def get_tracer():
    chis = np.linspace(CHIMIN, CHIMAX, NCHI)
    ws = np.ones(NCHI)
    t = ccl.Tracer()
    t.add_tracer(COSMO, kernel=(chis, ws))
    return t


@pytest.mark.parametrize("alpha,beta,typ", [(3., 3., 'SSC'),
                                            (3., 2., 'SSC'),
                                            (2., 3., 'SSC'),
                                            (2., 2., 'SSC'),
                                            (3., 3., 'cNG'),
                                            (2., 2., 'cNG'),
                                            (1., 2., 'cNG'),
                                            (2., 1., 'cNG')])
def test_cov_NG_sanity(alpha, beta, typ):
    # Compares covariance against analytical prediction
    tsp = get_tk3d(alpha, beta)
    tr = get_tracer()
    ls = np.array([2., 20., 200.])

    if typ == 'cNG':
        cov_p = pred_covar(ls[None, :], ls[:, None], alpha, beta)

        def cov_f(ll, **kwargs):
            return ccl.angular_cl_cov_cNG(COSMO, tr, tr, ll, tsp, **kwargs)
    elif typ == 'SSC':
        a_s = np.linspace(0.1, 1., 1024)
        s2b = np.ones_like(a_s)
        cov_p = pred_covar(ls[None, :], ls[:, None], alpha, beta,
                           prefac=1., chi_power=4)

        def cov_f(ll, **kwargs):
            return ccl.angular_cl_cov_SSC(COSMO, tr, tr, ll, tsp,
                                          sigma2_B=(a_s, s2b), **kwargs)

    cov = cov_f(ls)
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Spline integration (fast but inaccurate)
    cov = cov_f(ls, integration_method='spline')
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 4E-2)

    # Different tracers
    cov = cov_f(ls, cltracer3=tr, cltracer4=tr)
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Different ells
    cov = np.array([cov_f(ls, ell2=l)
                    for l in ls])
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # Different ells (transpose)
    cov = np.array([cov_f(l, ell2=ls)
                    for l in ls]).T
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)

    # All scalars
    cov = np.array([[cov_f(l1, ell2=l2)
                     for l1 in ls]
                    for l2 in ls])
    assert np.all(np.fabs(cov/cov_p-1).flatten() < 1E-5)


@pytest.mark.parametrize("typ", ['cNG', 'SSC'])
def test_cov_NG_errors(typ):
    if typ == 'cNG':
        cov_f = ccl.angular_cl_cov_cNG
    elif typ == 'SSC':
        cov_f = ccl.angular_cl_cov_SSC

    tsp = get_tk3d(1, 1)
    tr = get_tracer()
    ls = np.array([2., 20., 200.])

    assert_raises(ValueError, cov_f,
                  COSMO, tr, tr, ls, tsp,
                  integration_method='cag_cuad')

    assert_raises(ValueError, cov_f,
                  COSMO, tr, tr, ls, tr)


def test_Sigma2B():
    # Check projected variance calculation against
    # explicit integration
    from scipy.special import jv
    from scipy.integrate import simps

    fsky = 0.1
    # Default sampling
    a, s2b_a = ccl.sigma2_B_disc(COSMO, fsky=fsky)
    idx = (a > 0.5) & (a < 1)

    a_use = a[idx]
    # Input sampling
    s2b_b = ccl.sigma2_B_disc(COSMO, a=a_use, fsky=fsky)
    # Scalar input sampling
    s2b_c = np.array([ccl.sigma2_B_disc(COSMO, a=a, fsky=fsky)
                      for a in a_use])

    # Alternative calculation
    chis = ccl.comoving_radial_distance(COSMO, a_use)
    Rs = np.arccos(1-2*fsky)*chis

    def integrand(lk, a, R):
        k = np.exp(lk)
        x = k*R
        w = 2*jv(1, x)/x
        pk = ccl.linear_matter_power(COSMO, k, a)
        return w*w*k*k*pk/(2*np.pi)

    lk_arr = np.log(np.geomspace(1E-4, 1E1, 1024))
    s2b_d = np.array([simps(integrand(lk_arr, a, R), x=lk_arr)
                      for a, R in zip(a_use, Rs)])

    assert np.all(np.fabs(s2b_b/s2b_a[idx]-1) < 1E-10)
    assert np.all(np.fabs(s2b_c/s2b_a[idx]-1) < 1E-10)
    assert np.all(np.fabs(s2b_d/s2b_a[idx]-1) < 1E-3)

    # Check calculation based on mask Cls against the disc calculation
    ell = np.arange(1000)
    theta = np.arccos(1-2*fsky)
    kR = (ell+0.5)*theta
    mask_wl = (ell+0.5)/(2*np.pi) * (2*jv(1, kR)/(kR))**2

    a_use = np.array([0.2, 0.5, 1.0])
    s2b_e = ccl.sigma2_B_from_mask(COSMO, a=a_use, mask_wl=mask_wl)
    s2b_f = ccl.sigma2_B_disc(COSMO, a=a_use, fsky=fsky)
    assert np.all(np.fabs(s2b_e/s2b_f-1) < 1E-3)
