import os
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
import pytest


@pytest.fixture(scope='module', params=['fftlog', 'bessel'])
def corr_method(request):
    errfacs = {'fftlog': 0.22, 'bessel': 0.22}
    return request.param, errfacs[request.param]


@pytest.fixture(scope='module')
def set_up(request):
    dirdat = os.path.dirname(__file__) + '/data/'
    h0 = 0.70001831054687500
    logA = 3.05  # log(10^10 A_s)
    cosmo = ccl.Cosmology(Omega_c=0.12/h0**2, Omega_b=0.0221/h0**2, Omega_k=0,
                          h=h0, A_s=np.exp(logA)/10**10, n_s=0.96, Neff=3.046,
                          m_nu=0.0, w0=-1, wa=0, T_CMB=2.7255,
                          mu_0=0.1, sigma_0=0.1,
                          transfer_function='boltzmann_class',
                          matter_power_spectrum='linear')
    cosmo.cosmo.gsl_params.INTEGRATION_LIMBER_EPSREL = 2.5E-5
    cosmo.cosmo.gsl_params.INTEGRATION_EPSREL = 2.5E-5

    # Ell-dependent correction factors
    # Set up array of ells
    fl = {}
    lmax = 10000
    nls = (lmax - 400)//20+141
    ells = np.zeros(nls)
    ells[:101] = np.arange(101)
    ells[101:121] = ells[100] + (np.arange(20) + 1) * 5
    ells[121:141] = ells[120] + (np.arange(20) + 1) * 10
    ells[141:] = ells[140] + (np.arange(nls - 141) + 1) * 20
    fl['lmax'] = lmax
    fl['ells'] = ells

    # Load dNdz's
    z1, pz1 = np.loadtxt(dirdat + "bin1_histo.txt", unpack=True)
    z2, pz2 = np.loadtxt(dirdat + "bin2_histo.txt",  unpack=True)

    # Set up the linear galaxy bias as used in generating benchmarks
    bz1 = 1.45*np.ones_like(pz1)
    bz2 = 1.55*np.ones_like(pz2)

    # Initialize tracers
    trc = {}
    trc['g1'] = ccl.NumberCountsTracer(cosmo, False, (z1, pz1), (z1, bz1))
    trc['g2'] = ccl.NumberCountsTracer(cosmo, False, (z2, pz2), (z2, bz2))
    trc['l1'] = ccl.WeakLensingTracer(cosmo, (z1, pz1))
    trc['l2'] = ccl.WeakLensingTracer(cosmo, (z2, pz2))

    # Read benchmarks
    bms = {}
    bms['dd_11'] = np.loadtxt(dirdat+'/wtheta_mu_0p1_Sigma_0p1.dat')[0:15]
    bms['dd_22'] = np.loadtxt(dirdat+'/wtheta_mu_0p1_Sigma_0p1.dat')[15:]
    bms['dl_11'] = np.loadtxt(dirdat+'/gammat_mu_0p1_Sigma_0p1.dat')[0:15]
    bms['dl_12'] = np.loadtxt(dirdat+'/gammat_mu_0p1_Sigma_0p1.dat')[15:30]
    bms['dl_21'] = np.loadtxt(dirdat+'/gammat_mu_0p1_Sigma_0p1.dat')[30:45]
    bms['dl_22'] = np.loadtxt(dirdat+'/gammat_mu_0p1_Sigma_0p1.dat')[45:]
    bms['ll_11_p'] = np.loadtxt(dirdat+'/xip_mu_0p1_Sigma_0p1.dat')[0:14]
    bms['ll_12_p'] = np.loadtxt(dirdat+'/xip_mu_0p1_Sigma_0p1.dat')[14:28]
    bms['ll_22_p'] = np.loadtxt(dirdat+'/xip_mu_0p1_Sigma_0p1.dat')[28:]
    bms['ll_11_m'] = np.loadtxt(dirdat+'/xim_mu_0p1_Sigma_0p1.dat')[0:15]
    bms['ll_12_m'] = np.loadtxt(dirdat+'/xim_mu_0p1_Sigma_0p1.dat')[15:30]
    bms['ll_22_m'] = np.loadtxt(dirdat+'/xim_mu_0p1_Sigma_0p1.dat')[30:]
    theta = np.loadtxt(dirdat+'/theta_corr_MG.dat')
    bms['theta'] = theta

    # Read error bars
    ers = {}
    d = np.loadtxt("benchmarks/data/sigma_clustering_Nbin5",
                   unpack=True)
    ers['dd_11'] = interp1d(d[0], d[1],
                            fill_value=d[1][0],
                            bounds_error=False)(theta)
    ers['dd_22'] = interp1d(d[0], d[2],
                            fill_value=d[2][0],
                            bounds_error=False)(theta)
    d = np.loadtxt("benchmarks/data/sigma_ggl_Nbin5",
                   unpack=True)
    ers['dl_12'] = interp1d(d[0], d[1],
                            fill_value=d[1][0],
                            bounds_error=False)(theta)
    ers['dl_11'] = interp1d(d[0], d[2],
                            fill_value=d[2][0],
                            bounds_error=False)(theta)
    ers['dl_22'] = interp1d(d[0], d[3],
                            fill_value=d[3][0],
                            bounds_error=False)(theta)
    ers['dl_21'] = interp1d(d[0], d[4],
                            fill_value=d[4][0],
                            bounds_error=False)(theta)
    d = np.loadtxt("benchmarks/data/sigma_xi+_Nbin5",
                   unpack=True)
    # We cut the largest theta angle from xip because of issues
    # with the benchmark.
    ers['ll_11_p'] = interp1d(d[0], d[1],
                              fill_value=d[1][0],
                              bounds_error=False)(theta[0:14])
    ers['ll_22_p'] = interp1d(d[0], d[2],
                              fill_value=d[2][0],
                              bounds_error=False)(theta[0:14])
    ers['ll_12_p'] = interp1d(d[0], d[3],
                              fill_value=d[3][0],
                              bounds_error=False)(theta[0:14])
    d = np.loadtxt("benchmarks/data/sigma_xi-_Nbin5",
                   unpack=True)
    ers['ll_11_m'] = interp1d(d[0], d[1],
                              fill_value=d[1][0],
                              bounds_error=False)(theta)
    ers['ll_22_m'] = interp1d(d[0], d[2],
                              fill_value=d[2][0],
                              bounds_error=False)(theta)
    ers['ll_12_m'] = interp1d(d[0], d[3],
                              fill_value=d[3][0],
                              bounds_error=False)(theta)
    return cosmo, trc, bms, ers, fl


@pytest.mark.parametrize("t1,t2,bm,er,kind,pref",
                         [('g1', 'g1', 'dd_11', 'dd_11', 'NN', 1),
                          ('g2', 'g2', 'dd_22', 'dd_22', 'NN', 1),
                          ('g1', 'l1', 'dl_11', 'dl_11', 'NG', 1),
                          ('g1', 'l2', 'dl_12', 'dl_12', 'NG', 1),
                          ('g2', 'l1', 'dl_21', 'dl_21', 'NG', 1),
                          ('g2', 'l2', 'dl_22', 'dl_22', 'NG', 1),
                          ('l1', 'l1', 'll_11_p', 'll_11_p', 'GG+', 1),
                          ('l1', 'l2', 'll_12_p', 'll_12_p', 'GG+', 1),
                          ('l2', 'l2', 'll_22_p', 'll_22_p', 'GG+', 1),
                          ('l1', 'l1', 'll_11_m', 'll_11_m', 'GG-', 1),
                          ('l1', 'l2', 'll_12_m', 'll_12_m', 'GG-', 1),
                          ('l2', 'l2', 'll_22_m', 'll_22_m', 'GG-', 1)])
def test_xi(set_up, corr_method, t1, t2, bm, er, kind, pref):
    cosmo, trcs, bms, ers, fls = set_up
    method, errfac = corr_method

    # Debugging - define the  same cosmology but in GR

    cl = ccl.angular_cl(cosmo, trcs[t1], trcs[t2], fls['ells'])

    ell = np.arange(fls['lmax'])
    cli = interp1d(fls['ells'], cl, kind='cubic')(ell)
    # Our benchmarks have theta in arcmin
    # but CCL requires it in degrees:
    theta_deg = bms['theta'] / 60.
    # We cut the largest theta value for xi+ because of issues with the
    # benchmarks.
    if kind == 'GG+':
        xi = ccl.correlation(cosmo, ell, cli, theta_deg[0:14],
                             type=kind, method=method)
    else:
        xi = ccl.correlation(cosmo, ell, cli, theta_deg,
                             type=kind, method=method)
    xi *= pref

    assert np.all(np.fabs(xi - bms[bm]) < ers[er] * errfac)
