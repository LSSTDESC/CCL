import os
import time
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
import pytest

T0 = 0.0
T0_CLS = 0.0


@pytest.fixture(scope='module', params=['fftlog', 'bessel'])
def corr_method(request):
    errfacs = {'fftlog': 0.2, 'bessel': 0.1}
    return request.param, errfacs[request.param]


@pytest.fixture(scope='module', params=['analytic', 'histo'])
def set_up(request):
    t0 = time.time()
    nztyp = request.param
    dirdat = os.path.dirname(__file__) + '/data/'
    cosmo = ccl.Cosmology(Omega_c=0.30, Omega_b=0.00, Omega_g=0, Omega_k=0,
                          h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                          w0=-1, wa=0, T_CMB=2.7, transfer_function='bbks',
                          mass_function='tinker',
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

    # Initialize tracers
    if nztyp == 'analytic':
        # Analytic case
        zmean_1 = 1.0
        sigz_1 = 0.15
        zmean_2 = 1.5
        sigz_2 = 0.15
        z1, tmp_a1 = np.loadtxt(dirdat + "ia_amp_analytic_1.txt", unpack=True)
        z2, tmp_a2 = np.loadtxt(dirdat + "ia_amp_analytic_2.txt", unpack=True)
        pz1 = np.exp(-0.5 * ((z1 - zmean_1) / sigz_1)**2)
        pz2 = np.exp(-0.5 * ((z2 - zmean_2) / sigz_2)**2)
    elif nztyp == 'histo':
        # Histogram case
        z1, pz1 = np.loadtxt(dirdat + "bin1_histo.txt", unpack=True)[:, 1:]
        _, tmp_a1 = np.loadtxt(dirdat + "ia_amp_histo_1.txt", unpack=True)
        z2, pz2 = np.loadtxt(dirdat + "bin2_histo.txt",  unpack=True)[:, 1:]
        _, tmp_a2 = np.loadtxt(dirdat + "ia_amp_histo_2.txt", unpack=True)
    else:
        raise ValueError("Wrong Nz type " + nztyp)
    bz = np.ones_like(pz1)

    # Renormalize the IA amplitude to be consistent with A_IA
    D1 = ccl.growth_factor(cosmo, 1./(1+z1))
    D2 = ccl.growth_factor(cosmo, 1./(1+z2))
    rho_m = ccl.physical_constants.RHO_CRITICAL * cosmo['Omega_m']
    a1 = - tmp_a1 * D1 / (5e-14 * rho_m)
    a2 = - tmp_a2 * D2 / (5e-14 * rho_m)

    # Initialize tracers
    trc = {}
    trc['g1'] = ccl.NumberCountsTracer(cosmo, False,
                                       (z1, pz1),
                                       (z2, bz))
    trc['g2'] = ccl.NumberCountsTracer(cosmo, False,
                                       (z2, pz2),
                                       (z2, bz))
    trc['l1'] = ccl.WeakLensingTracer(cosmo, (z1, pz1))
    trc['l2'] = ccl.WeakLensingTracer(cosmo, (z2, pz2))
    trc['i1'] = ccl.WeakLensingTracer(cosmo, (z1, pz1),
                                      has_shear=False,
                                      ia_bias=(z1, a1))
    trc['i2'] = ccl.WeakLensingTracer(cosmo, (z2, pz2),
                                      has_shear=False,
                                      ia_bias=(z2, a2))
    trc['ct'] = ccl.CMBLensingTracer(cosmo, 1100.)

    # Read benchmarks
    def read_bm(fname):
        th, xi = np.loadtxt(fname, unpack=True)
        return th, xi

    pre = dirdat + 'run_'
    post = nztyp + "_log_wt_"
    bms = {}
    theta, bms['dd_11'] = read_bm(pre + 'b1b1' + post + 'dd.txt')
    _, bms['dd_12'] = read_bm(pre + 'b1b2' + post + 'dd.txt')
    _, bms['dd_22'] = read_bm(pre + 'b2b2' + post + 'dd.txt')
    _, bms['dl_11'] = read_bm(pre + 'b1b1' + post + 'dl.txt')
    _, bms['dl_12'] = read_bm(pre + 'b1b2' + post + 'dl.txt')
    _, bms['dl_21'] = read_bm(pre + 'b2b1' + post + 'dl.txt')
    _, bms['dl_22'] = read_bm(pre + 'b2b2' + post + 'dl.txt')
    _, bms['di_11'] = read_bm(pre + 'b1b1' + post + 'di.txt')
    _, bms['di_12'] = read_bm(pre + 'b1b2' + post + 'di.txt')
    _, bms['di_21'] = read_bm(pre + 'b2b1' + post + 'di.txt')
    _, bms['di_22'] = read_bm(pre + 'b2b2' + post + 'di.txt')
    _, bms['ll_11_p'] = read_bm(pre + 'b1b1' + post + 'll_pp.txt')
    _, bms['ll_12_p'] = read_bm(pre + 'b1b2' + post + 'll_pp.txt')
    _, bms['ll_22_p'] = read_bm(pre + 'b2b2' + post + 'll_pp.txt')
    _, bms['ll_11_m'] = read_bm(pre + 'b1b1' + post + 'll_mm.txt')
    _, bms['ll_12_m'] = read_bm(pre + 'b1b2' + post + 'll_mm.txt')
    _, bms['ll_22_m'] = read_bm(pre + 'b2b2' + post + 'll_mm.txt')
    _, bms['li_11_p'] = read_bm(pre + 'b1b1' + post + 'li_pp.txt')
    _, bms['li_12_p'] = read_bm(pre + 'b1b2' + post + 'li_pp.txt')
    _, bms['li_22_p'] = read_bm(pre + 'b2b2' + post + 'li_pp.txt')
    _, bms['li_11_m'] = read_bm(pre + 'b1b1' + post + 'li_mm.txt')
    _, bms['li_12_m'] = read_bm(pre + 'b1b2' + post + 'li_mm.txt')
    _, bms['li_22_m'] = read_bm(pre + 'b2b2' + post + 'li_mm.txt')
    _, bms['ii_11_p'] = read_bm(pre + 'b1b1' + post + 'ii_pp.txt')
    _, bms['ii_12_p'] = read_bm(pre + 'b1b2' + post + 'ii_pp.txt')
    _, bms['ii_22_p'] = read_bm(pre + 'b2b2' + post + 'ii_pp.txt')
    _, bms['ii_11_m'] = read_bm(pre + 'b1b1' + post + 'ii_mm.txt')
    _, bms['ii_12_m'] = read_bm(pre + 'b1b2' + post + 'ii_mm.txt')
    _, bms['ii_22_m'] = read_bm(pre + 'b2b2' + post + 'ii_mm.txt')
    bms['theta'] = theta

    # Read error bars
    ers = {}
    d = np.loadtxt("benchmarks/data/sigma_clustering_Nbin5",
                   unpack=True)
    ers['dd_11'] = interp1d(d[0] / 60., d[1],
                            fill_value=d[1][0],
                            bounds_error=False)(theta)
    ers['dd_22'] = interp1d(d[0] / 60., d[2],
                            fill_value=d[2][0],
                            bounds_error=False)(theta)
    d = np.loadtxt("benchmarks/data/sigma_ggl_Nbin5",
                   unpack=True)
    ers['dl_12'] = interp1d(d[0] / 60., d[1],
                            fill_value=d[1][0],
                            bounds_error=False)(theta)
    ers['dl_11'] = interp1d(d[0] / 60., d[2],
                            fill_value=d[2][0],
                            bounds_error=False)(theta)
    ers['dl_22'] = interp1d(d[0] / 60., d[3],
                            fill_value=d[3][0],
                            bounds_error=False)(theta)
    ers['dl_21'] = interp1d(d[0] / 60., d[4],
                            fill_value=d[4][0],
                            bounds_error=False)(theta)
    d = np.loadtxt("benchmarks/data/sigma_xi+_Nbin5",
                   unpack=True)
    ers['ll_11_p'] = interp1d(d[0] / 60., d[1],
                              fill_value=d[1][0],
                              bounds_error=False)(theta)
    ers['ll_22_p'] = interp1d(d[0] / 60., d[2],
                              fill_value=d[2][0],
                              bounds_error=False)(theta)
    ers['ll_12_p'] = interp1d(d[0] / 60., d[3],
                              fill_value=d[3][0],
                              bounds_error=False)(theta)
    d = np.loadtxt("benchmarks/data/sigma_xi-_Nbin5",
                   unpack=True)
    ers['ll_11_m'] = interp1d(d[0] / 60., d[1],
                              fill_value=d[1][0],
                              bounds_error=False)(theta)
    ers['ll_22_m'] = interp1d(d[0] / 60., d[2],
                              fill_value=d[2][0],
                              bounds_error=False)(theta)
    ers['ll_12_m'] = interp1d(d[0] / 60., d[3],
                              fill_value=d[3][0],
                              bounds_error=False)(theta)
    print('setup time:', time.time() - t0)
    return cosmo, trc, bms, ers, fl


# Commented out because we don't have this covariance
# ('g1', 'g2', 'dd_12', 'dd_12', 'gg', 1),
@pytest.mark.parametrize("t1,t2,bm,er,kind,pref",
                         [('g1', 'g1', 'dd_11', 'dd_11', 'gg', 1),
                          ('g2', 'g2', 'dd_22', 'dd_22', 'gg', 1),
                          ('g1', 'l1', 'dl_11', 'dl_11', 'gl', 1),
                          ('g1', 'l2', 'dl_12', 'dl_12', 'gl', 1),
                          ('g2', 'l1', 'dl_21', 'dl_21', 'gl', 1),
                          ('g2', 'l2', 'dl_22', 'dl_22', 'gl', 1),
                          ('g1', 'i1', 'di_11', 'dl_11', 'gl', 1),
                          ('g1', 'i2', 'di_12', 'dl_12', 'gl', 1),
                          ('g2', 'i1', 'di_21', 'dl_21', 'gl', 1),
                          ('g2', 'i2', 'di_22', 'dl_22', 'gl', 1),
                          ('l1', 'l1', 'll_11_p', 'll_11_p', 'l+', 1),
                          ('l1', 'l2', 'll_12_p', 'll_12_p', 'l+', 1),
                          ('l2', 'l2', 'll_22_p', 'll_22_p', 'l+', 1),
                          ('l1', 'l1', 'll_11_m', 'll_11_m', 'l-', 1),
                          ('l1', 'l2', 'll_12_m', 'll_12_m', 'l-', 1),
                          ('l2', 'l2', 'll_22_m', 'll_22_m', 'l-', 1),
                          ('i1', 'l1', 'li_11_p', 'll_11_p', 'l+', 2),
                          ('i1', 'l2', 'li_12_p', 'll_11_p', 'l+', 1),
                          ('i2', 'l2', 'li_22_p', 'll_22_p', 'l+', 2),
                          ('i1', 'l1', 'li_11_m', 'll_11_m', 'l-', 2),
                          ('i1', 'l2', 'li_12_m', 'll_11_m', 'l-', 1),
                          ('i2', 'l2', 'li_22_m', 'll_22_m', 'l-', 2),
                          ('i1', 'i1', 'ii_11_p', 'll_11_p', 'l+', 1),
                          ('i1', 'i2', 'ii_12_p', 'll_12_p', 'l+', 1),
                          ('i2', 'i2', 'ii_22_p', 'll_22_p', 'l+', 1),
                          ('i1', 'i1', 'ii_11_m', 'll_11_m', 'l-', 1),
                          ('i1', 'i2', 'ii_12_m', 'll_12_m', 'l-', 1),
                          ('i2', 'i2', 'ii_22_m', 'll_22_m', 'l-', 1)])
def test_xi(set_up, corr_method, t1, t2, bm, er, kind, pref):
    cosmo, trcs, bms, ers, fls = set_up
    method, errfac = corr_method

    global T0_CLS
    t0 = time.time()
    cl = ccl.angular_cl(cosmo, trcs[t1], trcs[t2], fls['ells'])
    T0_CLS += (time.time() - t0)

    ell = np.arange(fls['lmax'])
    cli = interp1d(fls['ells'], cl, kind='cubic')(ell)

    global T0
    t0 = time.time()
    xi = ccl.correlation(cosmo, ell, cli, bms['theta'],
                         corr_type=kind, method=method)
    T0 += (time.time() - t0)

    xi *= pref
    assert np.all(np.fabs(xi - bms[bm]) < ers[er] * errfac)

    print("time:", T0)
    print("time cls:", T0_CLS)
