import numpy as np
import pyccl as ccl
import os


def set_up():
    dirdat = os.path.dirname(__file__) + '/data/'
    cosmo = ccl.Cosmology(Omega_c=0.30, Omega_b=0.00, Omega_g=0, Omega_k=0,
                          h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                          w0=-1, wa=0, transfer_function='bbks',
                          mass_function='tinker',
                          matter_power_spectrum='linear')
    cosmo.cosmo.params.T_CMB = 2.7
    cosmo.cosmo.gsl_params.INTEGRATION_LIMBER_EPSREL = 1E-4
    cosmo.cosmo.gsl_params.INTEGRATION_EPSREL = 1E-4

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


def check_cls(typ):
    cosmo, arrs = set_up()

    # Generate tracers
    a = arrs[typ]
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
    ct = ccl.CMBLensingTracer(cosmo, 1100.)

    # Ell-dependent correction factors
    nls = 541
    ells = np.zeros(nls)
    ells[:50] = np.arange(50)+2
    ells[50:] = ells[49] + 6 * (np.arange(nls-50) + 1)
    fl_one = np.ones(nls)
    fl_dl = (ells + 0.5)**2/np.sqrt((ells + 2.) * (ells + 1.) *
                                    ells * (ells - 1.))
    fl_ll = fl_dl**2
    fl_lc = ells * (ells + 1)/np.sqrt((ells + 2.) * (ells + 1.) *
                                      ells * (ells - 1.))
    fl_li = 2 * fl_dl

    # Read benchmarks
    def read_bm(fname):
        _, cl = np.loadtxt(fname, unpack=True)
        return cl[ells.astype(int)]

    pre = os.path.dirname(__file__) + '/data/run_'
    post = typ + "_log_cl_"
    cl_dd_11 = read_bm(pre + 'b1b1' + post + 'dd.txt')
    cl_dd_12 = read_bm(pre + 'b1b2' + post + 'dd.txt')
    cl_dd_22 = read_bm(pre + 'b2b2' + post + 'dd.txt')
    cl_dl_11 = read_bm(pre + 'b1b1' + post + 'dl.txt')
    cl_dl_12 = read_bm(pre + 'b1b2' + post + 'dl.txt')
    cl_dl_21 = read_bm(pre + 'b2b1' + post + 'dl.txt')
    cl_dl_22 = read_bm(pre + 'b2b2' + post + 'dl.txt')
    cl_di_11 = read_bm(pre + 'b1b1' + post + 'di.txt')
    cl_di_12 = read_bm(pre + 'b1b2' + post + 'di.txt')
    cl_di_21 = read_bm(pre + 'b2b1' + post + 'di.txt')
    cl_di_22 = read_bm(pre + 'b2b2' + post + 'di.txt')
    cl_dc_1 = read_bm(pre + 'b1b1' + post + 'dc.txt')
    cl_dc_2 = read_bm(pre + 'b2b2' + post + 'dc.txt')
    cl_ll_11 = read_bm(pre + 'b1b1' + post + 'll.txt')
    cl_ll_12 = read_bm(pre + 'b1b2' + post + 'll.txt')
    cl_ll_22 = read_bm(pre + 'b2b2' + post + 'll.txt')
    cl_li_11 = read_bm(pre + 'b1b1' + post + 'li.txt')
    cl_li_22 = read_bm(pre + 'b2b2' + post + 'li.txt')
    cl_lc_1 = read_bm(pre + 'b1b1' + post + 'lc.txt')
    cl_lc_2 = read_bm(pre + 'b2b2' + post + 'lc.txt')
    cl_ii_11 = read_bm(pre + 'b1b1' + post + 'ii.txt')
    cl_ii_12 = read_bm(pre + 'b1b2' + post + 'ii.txt')
    cl_ii_22 = read_bm(pre + 'b2b2' + post + 'ii.txt')
    cl_cc = read_bm(pre + 'log_cl_cc.txt')

    # Check power spectra
    def compare_cls(cosmo, t1, t2, ls, cl_bm,
                    cl_a1b1, cl_a1b2, cl_a2b1, cl_a2b2, fl):
        cl = ccl.angular_cl(cosmo, t1, t2, ls) * fl
        el = np.sqrt((cl_a1b1 * cl_a2b2 + cl_a1b2 * cl_a2b1) / (2 * ls + 1.))
        assert np.all(np.fabs(cl - cl_bm) < 0.1 * el)

    # NC1-NC1
    compare_cls(cosmo, g1, g1, ells, cl_dd_11,
                cl_dd_11, cl_dd_11, cl_dd_11, cl_dd_11, fl_one)
    # NC1-NC2
    compare_cls(cosmo, g1, g2, ells, cl_dd_12,
                cl_dd_11, cl_dd_12, cl_dd_12, cl_dd_22, fl_one)
    # NC2-NC2
    compare_cls(cosmo, g2, g2, ells, cl_dd_22,
                cl_dd_22, cl_dd_22, cl_dd_22, cl_dd_22, fl_one)
    # NC1-WL1
    compare_cls(cosmo, g1, l1, ells, cl_dl_11,
                cl_dd_11, cl_dl_11, cl_dl_11, cl_ll_11, fl_dl)
    # NC1-WL2
    compare_cls(cosmo, g1, l2, ells, cl_dl_12,
                cl_dd_11, cl_dl_12, cl_dl_12, cl_ll_22, fl_dl)
    # NC2-WL1
    compare_cls(cosmo, g2, l1, ells, cl_dl_21,
                cl_dd_22, cl_dl_21, cl_dl_21, cl_ll_11, fl_dl)
    # NC2-WL2
    compare_cls(cosmo, g2, l2, ells, cl_dl_22,
                cl_dd_22, cl_dl_22, cl_dl_22, cl_ll_22, fl_dl)
    # NC1-IA1
    compare_cls(cosmo, g1, i1, ells, cl_di_11,
                cl_dd_11, cl_di_11, cl_di_11, cl_ii_11, fl_dl)
    # NC1-IA2
    compare_cls(cosmo, g1, i2, ells, cl_di_12,
                cl_dd_11, cl_di_12, cl_di_12, cl_ii_22, fl_dl)
    # NC2-IA1
    compare_cls(cosmo, g2, i1, ells, cl_di_21,
                cl_dd_22, cl_di_21, cl_di_21, cl_ii_11, fl_dl)
    # NC2-IA2
    compare_cls(cosmo, g2, i2, ells, cl_di_22,
                cl_dd_22, cl_di_22, cl_di_22, cl_ii_22, fl_dl)
    # NC1-CMBL
    compare_cls(cosmo, g1, ct, ells, cl_dc_1,
                cl_dd_11, cl_dc_1, cl_dc_1, cl_cc, fl_one)
    # NC2-CMBL
    compare_cls(cosmo, g2, ct, ells, cl_dc_2,
                cl_dd_22, cl_dc_2, cl_dc_2, cl_cc, fl_one)
    # WL1-WL1
    compare_cls(cosmo, l1, l1, ells, cl_ll_11,
                cl_ll_11, cl_ll_11, cl_ll_11, cl_ll_11, fl_ll)
    # WL1-WL2
    compare_cls(cosmo, l1, l2, ells, cl_ll_12,
                cl_ll_11, cl_ll_12, cl_ll_12, cl_ll_22, fl_ll)
    # WL2-WL2
    compare_cls(cosmo, l2, l2, ells, cl_ll_22,
                cl_ll_22, cl_ll_22, cl_ll_22, cl_ll_22, fl_ll)
    # WL1-IA1
    compare_cls(cosmo, l1, i1, ells, cl_li_11,
                cl_ii_11, cl_li_11, cl_li_11, cl_ll_11, fl_li)
    # WL2-IA2
    compare_cls(cosmo, l2, i2, ells, cl_li_22,
                cl_ii_22, cl_li_22, cl_li_22, cl_ll_22, fl_li)
    # WL1-CMBL
    compare_cls(cosmo, l1, ct, ells, cl_lc_1,
                cl_ll_11, cl_lc_1, cl_lc_1, cl_cc, fl_lc)
    # WL2-CMBL
    compare_cls(cosmo, l2, ct, ells, cl_lc_2,
                cl_ll_22, cl_lc_2, cl_lc_2, cl_cc, fl_lc)
    # IA1-IA1
    compare_cls(cosmo, i1, i1, ells, cl_ii_11,
                cl_ll_11, cl_ll_11, cl_ll_11, cl_ll_11, fl_ll)
    # IA1-IA2
    compare_cls(cosmo, i1, i2, ells, cl_ii_12,
                cl_ll_11, cl_ll_12, cl_ll_12, cl_ll_22, fl_ll)
    # IA2-IA2
    compare_cls(cosmo, i2, i2, ells, cl_ii_22,
                cl_ll_22, cl_ll_22, cl_ll_22, cl_ll_22, fl_ll)


def test_cls_analytic():
    check_cls('analytic')


def test_cls_histo():
    check_cls('histo')
