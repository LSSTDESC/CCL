import os
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d


def set_up():
    dirdat = os.path.dirname(__file__) + '/data/'
    cosmo = ccl.Cosmology(Omega_c=0.30, Omega_b=0.00, Omega_g=0, Omega_k=0,
                          h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                          w0=-1, wa=0, transfer_function='bbks',
                          mass_function='tinker',
                          matter_power_spectrum='linear')
    cosmo.cosmo.params.T_CMB = 2.7
    cosmo.cosmo.gsl_params.INTEGRATION_LIMBER_EPSREL = 2.5E-5
    cosmo.cosmo.gsl_params.INTEGRATION_EPSREL = 2.5E-5

    arrays = {}
    # Analytic case
    arrays['analytic'] = {}
    zmean_1 = 1.0
    sigz_1 = 0.15
    zmean_2 = 1.5
    sigz_2 = 0.15
    z1, a1 = np.loadtxt(dirdat + "ia_amp_analytic_1.txt", unpack=True)
    z2, a2 = np.loadtxt(dirdat + "ia_amp_analytic_2.txt", unpack=True)
    pz1 = np.exp(-0.5 * ((z1 - zmean_1) / sigz_1)**2)
    pz2 = np.exp(-0.5 * ((z2 - zmean_2) / sigz_2)**2)
    arrays['analytic']['z1'] = z1
    arrays['analytic']['z2'] = z2
    arrays['analytic']['a1'] = a1
    arrays['analytic']['a2'] = a2
    arrays['analytic']['p1'] = pz1
    arrays['analytic']['p2'] = pz2
    arrays['analytic']['bz'] = np.ones_like(pz1)
    arrays['analytic']['rz'] = np.ones_like(pz1)

    # Histogram case
    arrays['histo'] = {}
    z1, pz1 = np.loadtxt(dirdat + "bin1_histo.txt", unpack=True)[:, 1:]
    zz1, a1 = np.loadtxt(dirdat + "ia_amp_histo_1.txt", unpack=True)
    z2, pz2 = np.loadtxt(dirdat + "bin2_histo.txt",  unpack=True)[:, 1:]
    _, a2 = np.loadtxt(dirdat + "ia_amp_histo_2.txt", unpack=True)
    arrays['histo']['z1'] = z1
    arrays['histo']['z2'] = z2
    arrays['histo']['a1'] = a1
    arrays['histo']['a2'] = a2
    arrays['histo']['p1'] = pz1
    arrays['histo']['p2'] = pz2
    arrays['histo']['bz'] = np.ones_like(pz1)
    arrays['histo']['rz'] = np.ones_like(pz1)

    return cosmo, arrays


def check_cls(nztyp, corr_method, error_fraction):
    cosmo, arrs = set_up()

    # Generate tracers
    a = arrs[nztyp]
    g1 = ccl.NumberCountsTracer(cosmo, False,
                                (a['z1'], a['p1']),
                                (a['z1'], a['bz']))
    g2 = ccl.NumberCountsTracer(cosmo, False,
                                (a['z2'], a['p2']),
                                (a['z2'], a['bz']))
    l1 = ccl.WeakLensingTracer(cosmo, (a['z1'], a['p1']))
    l2 = ccl.WeakLensingTracer(cosmo, (a['z2'], a['p2']))
    i1 = ccl.WeakLensingTracer(cosmo, (a['z1'], a['p1']),
                               has_shear=False,
                               ia_bias=(a['z1'], a['a1']),
                               red_frac=(a['z1'], a['rz']))
    i2 = ccl.WeakLensingTracer(cosmo, (a['z2'], a['p2']),
                               has_shear=False,
                               ia_bias=(a['z2'], a['a2']),
                               red_frac=(a['z2'], a['rz']))

    # Ell-dependent correction factors
    # Set up array of ells
    lmax = 10000
    nls = (lmax - 400)//20+141
    ells = np.zeros(nls)
    ells[:101] = np.arange(101)
    ells[101:121] = ells[100] + (np.arange(20) + 1) * 5
    ells[121:141] = ells[120] + (np.arange(20) + 1) * 10
    ells[141:] = ells[140] + (np.arange(nls - 141) + 1) * 20

    # Read benchmarks
    def read_bm(fname):
        th, xi = np.loadtxt(fname, unpack=True)
        return th, xi

    pre = os.path.dirname(__file__) + '/data/run_'
    post = nztyp + "_log_wt_"
    theta, xi_dd_11 = read_bm(pre + 'b1b1' + post + 'dd.txt')
    _, xi_dd_12 = read_bm(pre + 'b1b2' + post + 'dd.txt')
    _, xi_dd_22 = read_bm(pre + 'b2b2' + post + 'dd.txt')
    _, xi_dl_11 = read_bm(pre + 'b1b1' + post + 'dl.txt')
    _, xi_dl_12 = read_bm(pre + 'b1b2' + post + 'dl.txt')
    _, xi_dl_21 = read_bm(pre + 'b2b1' + post + 'dl.txt')
    _, xi_dl_22 = read_bm(pre + 'b2b2' + post + 'dl.txt')
    _, xi_di_11 = read_bm(pre + 'b1b1' + post + 'di.txt')
    _, xi_di_12 = read_bm(pre + 'b1b2' + post + 'di.txt')
    _, xi_di_21 = read_bm(pre + 'b2b1' + post + 'di.txt')
    _, xi_di_22 = read_bm(pre + 'b2b2' + post + 'di.txt')
    _, xi_ll_11_p = read_bm(pre + 'b1b1' + post + 'll_pp.txt')
    _, xi_ll_12_p = read_bm(pre + 'b1b2' + post + 'll_pp.txt')
    _, xi_ll_22_p = read_bm(pre + 'b2b2' + post + 'll_pp.txt')
    _, xi_ll_11_m = read_bm(pre + 'b1b1' + post + 'll_mm.txt')
    _, xi_ll_12_m = read_bm(pre + 'b1b2' + post + 'll_mm.txt')
    _, xi_ll_22_m = read_bm(pre + 'b2b2' + post + 'll_mm.txt')
    _, xi_li_11_p = read_bm(pre + 'b1b1' + post + 'li_pp.txt')
    _, xi_li_12_p = read_bm(pre + 'b1b2' + post + 'li_pp.txt')
    _, xi_li_22_p = read_bm(pre + 'b2b2' + post + 'li_pp.txt')
    _, xi_li_11_m = read_bm(pre + 'b1b1' + post + 'li_mm.txt')
    _, xi_li_12_m = read_bm(pre + 'b1b2' + post + 'li_mm.txt')
    _, xi_li_22_m = read_bm(pre + 'b2b2' + post + 'li_mm.txt')
    _, xi_ii_11_p = read_bm(pre + 'b1b1' + post + 'ii_pp.txt')
    _, xi_ii_12_p = read_bm(pre + 'b1b2' + post + 'ii_pp.txt')
    _, xi_ii_22_p = read_bm(pre + 'b2b2' + post + 'ii_pp.txt')
    _, xi_ii_11_m = read_bm(pre + 'b1b1' + post + 'ii_mm.txt')
    _, xi_ii_12_m = read_bm(pre + 'b1b2' + post + 'ii_mm.txt')
    _, xi_ii_22_m = read_bm(pre + 'b2b2' + post + 'ii_mm.txt')

    # Read error bars
    d = np.loadtxt("tests/benchmark/cov_corr/sigma_clustering_Nbin5",
                   unpack=True)
    exi_dd_11 = interp1d(d[0] / 60., d[1],
                         fill_value=d[1][0],
                         bounds_error=False)(theta)
    exi_dd_22 = interp1d(d[0] / 60., d[2],
                         fill_value=d[2][0],
                         bounds_error=False)(theta)
    d = np.loadtxt("tests/benchmark/cov_corr/sigma_ggl_Nbin5",
                   unpack=True)
    exi_dl_12 = interp1d(d[0] / 60., d[1],
                         fill_value=d[1][0],
                         bounds_error=False)(theta)
    exi_dl_11 = interp1d(d[0] / 60., d[2],
                         fill_value=d[2][0],
                         bounds_error=False)(theta)
    exi_dl_22 = interp1d(d[0] / 60., d[3],
                         fill_value=d[3][0],
                         bounds_error=False)(theta)
    exi_dl_21 = interp1d(d[0] / 60., d[4],
                         fill_value=d[4][0],
                         bounds_error=False)(theta)
    d = np.loadtxt("tests/benchmark/cov_corr/sigma_xi+_Nbin5",
                   unpack=True)
    exi_ll_11_p = interp1d(d[0] / 60., d[1],
                           fill_value=d[1][0],
                           bounds_error=False)(theta)
    exi_ll_22_p = interp1d(d[0] / 60., d[2],
                           fill_value=d[2][0],
                           bounds_error=False)(theta)
    exi_ll_12_p = interp1d(d[0] / 60., d[3],
                           fill_value=d[3][0],
                           bounds_error=False)(theta)
    d = np.loadtxt("tests/benchmark/cov_corr/sigma_xi-_Nbin5",
                   unpack=True)
    exi_ll_11_m = interp1d(d[0] / 60., d[1],
                           fill_value=d[1][0],
                           bounds_error=False)(theta)
    exi_ll_22_m = interp1d(d[0] / 60., d[2],
                           fill_value=d[2][0],
                           bounds_error=False)(theta)
    exi_ll_12_m = interp1d(d[0] / 60., d[3],
                           fill_value=d[3][0],
                           bounds_error=False)(theta)

    # Check power spectra
    def compare_xis(cosmo, t1, t2, ls, th,
                    xi_bm, e_xi, typ, method, prefactor=1):
        cl = ccl.angular_cl(cosmo, t1, t2, ls)
        ell = np.arange(lmax)
        cli = interp1d(ls, cl, kind='cubic')(ell)
        xi = ccl.correlation(cosmo, ell, cli, th,
                             corr_type=typ, method=method)
        xi *= prefactor
        assert np.all(np.fabs(xi-xi_bm) < e_xi * error_fraction)

    # NC1-NC1
    compare_xis(cosmo, g1, g1, ells, theta, xi_dd_11,
                exi_dd_11, 'gg', corr_method)
    # NC1-NC2
    # Commented out because we do not currently have the covariance.
    # compare_xis(cosmo, g1, g2, ells, theta, xi_dd_12,
    #             exi_dd_12, 'gg', corr_method)
    # NC2-NC2
    compare_xis(cosmo, g2, g2, ells, theta, xi_dd_22,
                exi_dd_22, 'gg', corr_method)
    # NC1-WL1
    compare_xis(cosmo, g1, l1, ells, theta, xi_dl_11,
                exi_dl_11, 'gl', corr_method)
    # NC1-WL2
    compare_xis(cosmo, g1, l2, ells, theta, xi_dl_12,
                exi_dl_12, 'gl', corr_method)
    # NC2-WL1
    compare_xis(cosmo, g2, l1, ells, theta, xi_dl_21,
                exi_dl_21, 'gl', corr_method)
    # NC2-WL2
    compare_xis(cosmo, g2, l2, ells, theta, xi_dl_22,
                exi_dl_22, 'gl', corr_method)
    # NC1-IA1
    compare_xis(cosmo, g1, i1, ells, theta, xi_di_11,
                exi_dl_11, 'gl', corr_method)
    # NC1-IA2
    compare_xis(cosmo, g1, i2, ells, theta, xi_di_12,
                exi_dl_12, 'gl', corr_method)
    # NC2-IA1
    compare_xis(cosmo, g2, i1, ells, theta, xi_di_21,
                exi_dl_21, 'gl', corr_method)
    # NC2-IA2
    compare_xis(cosmo, g2, i2, ells, theta, xi_di_22,
                exi_dl_22, 'gl', corr_method)
    # WL1-WL1, +
    compare_xis(cosmo, l1, l1, ells, theta, xi_ll_11_p,
                exi_ll_11_p, 'l+', corr_method)
    # WL1-WL2, +
    compare_xis(cosmo, l1, l2, ells, theta, xi_ll_12_p,
                exi_ll_12_p, 'l+', corr_method)
    # WL2-WL2, +
    compare_xis(cosmo, l2, l2, ells, theta, xi_ll_22_p,
                exi_ll_22_p, 'l+', corr_method)
    # WL1-WL1, -
    compare_xis(cosmo, l1, l1, ells, theta, xi_ll_11_m,
                exi_ll_11_m, 'l-', corr_method)
    # WL1-WL2, -
    compare_xis(cosmo, l1, l2, ells, theta, xi_ll_12_m,
                exi_ll_12_m, 'l-', corr_method)
    # WL2-WL2, -
    compare_xis(cosmo, l2, l2, ells, theta, xi_ll_22_m,
                exi_ll_22_m, 'l-', corr_method)
    # IA1-WL1, +
    compare_xis(cosmo, i1, l1, ells, theta, xi_li_11_p,
                exi_ll_11_p, 'l+', corr_method, prefactor=2)
    # IA1-WL2, +
    compare_xis(cosmo, i1, l2, ells, theta, xi_li_12_p,
                exi_ll_11_p, 'l+', corr_method)
    # IA2-WL2, +
    compare_xis(cosmo, i2, l2, ells, theta, xi_li_22_p,
                exi_ll_22_p, 'l+', corr_method, prefactor=2)
    # IA1-WL1, -
    compare_xis(cosmo, i1, l1, ells, theta, xi_li_11_m,
                exi_ll_11_m, 'l-', corr_method, prefactor=2)
    # IA1-WL2, -
    compare_xis(cosmo, i1, l2, ells, theta, xi_li_12_m,
                exi_ll_11_m, 'l-', corr_method)
    # IA2-WL2, -
    compare_xis(cosmo, i2, l2, ells, theta, xi_li_22_m,
                exi_ll_22_m, 'l-', corr_method, prefactor=2)
    # IA1-IA1, +
    compare_xis(cosmo, i1, i1, ells, theta, xi_ii_11_p,
                exi_ll_11_p, 'l+', corr_method)
    # IA1-IA2, +
    compare_xis(cosmo, i1, i2, ells, theta, xi_ii_12_p,
                exi_ll_11_p, 'l+', corr_method)
    # IA2-IA2, +
    compare_xis(cosmo, i2, i2, ells, theta, xi_ii_22_p,
                exi_ll_22_p, 'l+', corr_method)
    # IA1-IA1, -
    compare_xis(cosmo, i1, i1, ells, theta, xi_ii_11_m,
                exi_ll_11_m, 'l-', corr_method)
    # IA1-IA2, -
    compare_xis(cosmo, i1, i2, ells, theta, xi_ii_12_m,
                exi_ll_11_m, 'l-', corr_method)
    # IA2-IA2, -
    compare_xis(cosmo, i2, i2, ells, theta, xi_ii_22_m,
                exi_ll_22_m, 'l-', corr_method)


def test_xi_analytic_fftlog():
    check_cls('analytic', 'fftlog', 0.2)


def test_xi_histo_fftlog():
    check_cls('histo', 'fftlog', 0.2)


def test_xi_analytic_bessel():
    check_cls('analytic', 'bessel', 0.1)


def test_xi_histo_bessel():
    check_cls('histo', 'bessel', 0.1)
