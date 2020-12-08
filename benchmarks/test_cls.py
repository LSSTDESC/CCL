import numpy as np
import pyccl as ccl
import time
import os
import pytest


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
    cosmo.cosmo.gsl_params.INTEGRATION_LIMBER_EPSREL = 1E-4
    cosmo.cosmo.gsl_params.INTEGRATION_EPSREL = 1E-4

    # ell-arrays
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
    lfacs = {'ells': ells,
             'fl_one': fl_one,
             'fl_dl': fl_dl,
             'fl_ll': fl_ll,
             'fl_lc': fl_lc,
             'fl_li': fl_li}

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
        z2, pz2 = np.loadtxt(dirdat + "bin2_histo.txt", unpack=True)[:, 1:]
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
        _, cl = np.loadtxt(fname, unpack=True)
        return cl[ells.astype('int')]

    pre = dirdat + 'run_'
    post = nztyp + "_log_cl_"
    bms = {}
    bms['dd_11'] = read_bm(pre + 'b1b1' + post + 'dd.txt')
    bms['dd_12'] = read_bm(pre + 'b1b2' + post + 'dd.txt')
    bms['dd_22'] = read_bm(pre + 'b2b2' + post + 'dd.txt')
    bms['dl_11'] = read_bm(pre + 'b1b1' + post + 'dl.txt')
    bms['dl_12'] = read_bm(pre + 'b1b2' + post + 'dl.txt')
    bms['dl_21'] = read_bm(pre + 'b2b1' + post + 'dl.txt')
    bms['dl_22'] = read_bm(pre + 'b2b2' + post + 'dl.txt')
    bms['di_11'] = read_bm(pre + 'b1b1' + post + 'di.txt')
    bms['di_12'] = read_bm(pre + 'b1b2' + post + 'di.txt')
    bms['di_21'] = read_bm(pre + 'b2b1' + post + 'di.txt')
    bms['di_22'] = read_bm(pre + 'b2b2' + post + 'di.txt')
    bms['dc_1'] = read_bm(pre + 'b1b1' + post + 'dc.txt')
    bms['dc_2'] = read_bm(pre + 'b2b2' + post + 'dc.txt')
    bms['ll_11'] = read_bm(pre + 'b1b1' + post + 'll.txt')
    bms['ll_12'] = read_bm(pre + 'b1b2' + post + 'll.txt')
    bms['ll_22'] = read_bm(pre + 'b2b2' + post + 'll.txt')
    bms['li_11'] = read_bm(pre + 'b1b1' + post + 'li.txt')
    bms['li_22'] = read_bm(pre + 'b2b2' + post + 'li.txt')
    bms['lc_1'] = read_bm(pre + 'b1b1' + post + 'lc.txt')
    bms['lc_2'] = read_bm(pre + 'b2b2' + post + 'lc.txt')
    bms['ii_11'] = read_bm(pre + 'b1b1' + post + 'ii.txt')
    bms['ii_12'] = read_bm(pre + 'b1b2' + post + 'ii.txt')
    bms['ii_22'] = read_bm(pre + 'b2b2' + post + 'ii.txt')
    bms['cc'] = read_bm(pre + 'log_cl_cc.txt')
    print('init and i/o time:', time.time() - t0)

    return cosmo, trc, lfacs, bms


@pytest.mark.parametrize("t1,t2,bm,a1b1,a1b2,a2b1,a2b2,fl",
                         [('g1', 'g1', 'dd_11',   # NC1-NC1
                           'dd_11', 'dd_11', 'dd_11', 'dd_11',
                           'fl_one'),
                          ('g1', 'g2', 'dd_12',   # NC1-NC2
                           'dd_11', 'dd_12', 'dd_12', 'dd_22',
                           'fl_one'),
                          ('g2', 'g2', 'dd_22',   # NC2-NC2
                           'dd_22', 'dd_22', 'dd_22', 'dd_22',
                           'fl_one'),
                          ('g1', 'l1', 'dl_11',   # NC1-WL1
                           'dd_11', 'dl_11', 'dl_11', 'll_11',
                           'fl_dl'),
                          ('g1', 'l2', 'dl_12',   # NC1-WL2
                           'dd_11', 'dl_12', 'dl_12', 'll_22',
                           'fl_dl'),
                          ('g2', 'l1', 'dl_21',   # NC2-WL1
                           'dd_22', 'dl_21', 'dl_21', 'll_22',
                           'fl_dl'),
                          ('g2', 'l2', 'dl_22',   # NC2-WL2
                           'dd_22', 'dl_22', 'dl_22', 'ii_22',
                           'fl_dl'),
                          ('g1', 'i1', 'di_11',   # NC1-IA1
                           'dd_11', 'di_11', 'di_11', 'ii_11',
                           'fl_dl'),
                          ('g1', 'i2', 'di_12',   # NC1-IA2
                           'dd_11', 'di_12', 'di_12', 'ii_22',
                           'fl_dl'),
                          ('g2', 'i1', 'di_21',   # NC2-IA1
                           'dd_22', 'di_21', 'di_21', 'ii_22',
                           'fl_dl'),
                          ('g2', 'i2', 'di_22',   # NC2-IA2
                           'dd_22', 'di_22', 'di_22', 'ii_22',
                           'fl_dl'),
                          ('g1', 'ct', 'dc_1',   # NC1-CMBL
                           'dd_11', 'dc_1', 'dc_1', 'cc',
                           'fl_one'),
                          ('g2', 'ct', 'dc_2',   # NC2-CMBL
                           'dd_22', 'dc_2', 'dc_2', 'cc',
                           'fl_one'),
                          ('l1', 'l1', 'll_11',   # WL1-WL1
                           'll_11', 'll_11', 'll_11', 'll_11',
                           'fl_ll'),
                          ('l1', 'l2', 'll_12',   # WL1-WL2
                           'll_11', 'll_12', 'll_12', 'll_22',
                           'fl_ll'),
                          ('l2', 'l2', 'll_22',   # WL2-WL2
                           'll_22', 'll_22', 'll_22', 'll_22',
                           'fl_ll'),
                          ('l1', 'i1', 'li_11',   # WL1-IA1
                           'ii_11', 'li_11', 'li_11', 'll_11',
                           'fl_li'),
                          ('l2', 'i2', 'li_22',   # WL2-IA2
                           'ii_22', 'li_22', 'li_22', 'll_22',
                           'fl_li'),
                          ('l1', 'ct', 'lc_1',   # WL1-CMBL
                           'll_11', 'lc_1', 'lc_1', 'cc',
                           'fl_lc'),
                          ('l2', 'ct', 'lc_2',   # WL2-CMBL
                           'll_22', 'lc_2', 'lc_2', 'cc',
                           'fl_lc'),
                          ('i1', 'i1', 'ii_11',   # IA1-IA1
                           'll_11', 'll_11', 'll_11', 'll_11',
                           'fl_ll'),
                          ('i1', 'i2', 'ii_12',   # IA1-IA2
                           'll_11', 'll_12', 'll_12', 'll_22',
                           'fl_ll'),
                          ('i2', 'i2', 'ii_22',   # IA2-IA2
                           'll_22', 'll_22', 'll_22', 'll_22',
                           'fl_ll')])
def test_cls(set_up, t1, t2, bm,
             a1b1, a1b2, a2b1, a2b2, fl):
    cosmo, trcs, lfc, bmk = set_up
    cl = ccl.angular_cl(cosmo, trcs[t1], trcs[t2], lfc['ells'],
                        limber_integration_method='qag_quad') * lfc[fl]
    el = np.sqrt((bmk[a1b1] * bmk[a2b2] + bmk[a1b2] * bmk[a2b1]) /
                 (2 * lfc['ells'] + 1.))
    assert np.all(np.fabs(cl - bmk[bm]) < 0.1 * el)


@pytest.mark.parametrize("t1,t2,bm,a1b1,a1b2,a2b1,a2b2,fl",
                         [('g1', 'g1', 'dd_11',   # NC1-NC1
                           'dd_11', 'dd_11', 'dd_11', 'dd_11',
                           'fl_one'),
                          ('g1', 'g2', 'dd_12',   # NC1-NC2
                           'dd_11', 'dd_12', 'dd_12', 'dd_22',
                           'fl_one'),
                          ('l2', 'l2', 'll_22',   # WL2-WL2
                           'll_22', 'll_22', 'll_22', 'll_22',
                           'fl_ll')])
def test_cls_spline(set_up, t1, t2, bm,
                    a1b1, a1b2, a2b1, a2b2, fl):
    cosmo, trcs, lfc, bmk = set_up
    cl = ccl.angular_cl(cosmo, trcs[t1], trcs[t2], lfc['ells'],
                        limber_integration_method='spline') * lfc[fl]
    el = np.sqrt((bmk[a1b1] * bmk[a2b2] + bmk[a1b2] * bmk[a2b1]) /
                 (2 * lfc['ells'] + 1.))
    assert np.all(np.fabs(cl - bmk[bm]) < 0.2 * el)
