import os
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
import pytest


@pytest.fixture(scope='module', params=['fftlog', 'bessel'])
def corr_method(request):
    errfacs = {'fftlog': 0.2, 'bessel': 0.1}
    return request.param, errfacs[request.param]


@pytest.fixture(scope='module', params=['analytic', 'histo'])
def set_up(request):
    nztyp = request.param
    dirdat = os.path.dirname(__file__) + '/data/'
    h0 = 0.67192993164062498
    logA = 3.05 # log(10^10 A_s)
    # DL: current benchmarks still have mnu = 0.06, need to be rerun with mnu=0.
    cosmo = ccl.Cosmology(Omega_c=0.12/h0**2, Omega_b=0.0221/h0**2, Omega_k=0,
                          h=h0, A_s = np.exp(logA)/10**10, n_s=0.96, Neff=3.046, m_nu=0.0,
                          w0=-1, wa=0, transfer_function='boltzmann_class',
                          matter_power_spectrum='linear')
    # DL: Here T_CMB is being overridden because it is hardcoded.
    # Check what T_CMB is for our code and if this is necessary.                      
    cosmo.cosmo.params.T_CMB = 2.7
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
    (zlow, zg, zhigh, pz1_g, pz2_g, 
     pz3_g, pz4_g, pz5_g) = np.loadtxt(dirdat + "nz_lens_corr_MG.dat", 
                                          unpack=True)
    (zlow, zl, zhigh, pz1_l, pz2_l, 
     pz3_l, pz4_l) = np.loadtxt(dirdat + "nz_src_corr_MG.dat", 
                                          unpack=True)
    
    # DL: bias needs to match what is used in getting benchmarks. What
    # is that? I don't see it in the params file.                                       
    bz = np.ones_like(pz1_g)
    rz = np.ones_like(pz1_g)

    # Initialize tracers
    trc = {}
    trc['g1'] = ccl.NumberCountsTracer(cosmo, False,
                                       (zg, pz1_g),
                                       (zg, bz))
    trc['g2'] = ccl.NumberCountsTracer(cosmo, False,
                                       (zg, pz2_g),
                                       (zg, bz))
    trc['g3'] = ccl.NumberCountsTracer(cosmo, False,
                                       (zg, pz3_g),
                                       (zg, bz))
    trc['g4'] = ccl.NumberCountsTracer(cosmo, False,
                                       (zg, pz4_g),
                                       (zg, bz))
    trc['g5'] = ccl.NumberCountsTracer(cosmo, False,
                                       (zg, pz5_g),
                                       (zg, bz))                                                                                                         
                                       
    trc['l1'] = ccl.WeakLensingTracer(cosmo, (zl, pz1_l))
    trc['l2'] = ccl.WeakLensingTracer(cosmo, (zl, pz2_l))
    trc['l3'] = ccl.WeakLensingTracer(cosmo, (zl, pz3_l))
    trc['l4'] = ccl.WeakLensingTracer(cosmo, (zl, pz4_l))

    # Read benchmarks
    def read_bm(fname):
        th, xi = np.loadtxt(fname, unpack=True)
        return th, xi

    # DL: need to make sure the names are correct.
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
    _, bms['ll_11_p'] = read_bm(pre + 'b1b1' + post + 'll_pp.txt')
    _, bms['ll_12_p'] = read_bm(pre + 'b1b2' + post + 'll_pp.txt')
    _, bms['ll_22_p'] = read_bm(pre + 'b2b2' + post + 'll_pp.txt')
    _, bms['ll_11_m'] = read_bm(pre + 'b1b1' + post + 'll_mm.txt')
    _, bms['ll_12_m'] = read_bm(pre + 'b1b2' + post + 'll_mm.txt')
    _, bms['ll_22_m'] = read_bm(pre + 'b2b2' + post + 'll_mm.txt')
    bms['theta'] = theta

    # Read error bars
    # DL: need to make sure these match what we are using.
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
    return cosmo, trc, bms, ers, fl

# DL probably need to change this back to less bins to match where
# we have error calculations.
@pytest.mark.parametrize("t1,t2,bm,er,kind,pref",
                         [('g1', 'g1', 'dd_11', 'dd_11', 'gg', 1),
                          ('g2', 'g2', 'dd_22', 'dd_22', 'gg', 1),
                          ('g3', 'g3', 'dd_33', 'dd_33', 'gg', 1),
                          ('g4', 'g4', 'dd_44', 'dd_44', 'gg', 1),
                          ('g5', 'g5', 'dd_55', 'dd_55', 'gg', 1),
                          ('g1', 'l1', 'dl_11', 'dl_11', 'gl', 1),
                          ('g1', 'l2', 'dl_12', 'dl_12', 'gl', 1),
                          ('g1', 'l3', 'dl_13', 'dl_13', 'gl', 1),
                          ('g1', 'l4', 'dl_14', 'dl_14', 'gl', 1),
                          ('g2', 'l1', 'dl_21', 'dl_21', 'gl', 1),
                          ('g2', 'l2', 'dl_22', 'dl_22', 'gl', 1),
                          ('g2', 'l3', 'dl_23', 'dl_23', 'gl', 1),
                          ('g2', 'l4', 'dl_24', 'dl_24', 'gl', 1),
                          ('g3', 'l1', 'dl_31', 'dl_31', 'gl', 1),
                          ('g3', 'l2', 'dl_32', 'dl_32', 'gl', 1),
                          ('g3', 'l3', 'dl_33', 'dl_33', 'gl', 1),
                          ('g3', 'l4', 'dl_34', 'dl_34', 'gl', 1),
                          ('g4', 'l1', 'dl_41', 'dl_41', 'gl', 1),
                          ('g4', 'l2', 'dl_42', 'dl_42', 'gl', 1),
                          ('g4', 'l3', 'dl_43', 'dl_43', 'gl', 1),
                          ('g4', 'l4', 'dl_44', 'dl_44', 'gl', 1),
                          ('g5', 'l1', 'dl_51', 'dl_51', 'gl', 1),
                          ('g5', 'l2', 'dl_52', 'dl_52', 'gl', 1),
                          ('g5', 'l3', 'dl_53', 'dl_53', 'gl', 1),
                          ('g5', 'l4', 'dl_54', 'dl_54', 'gl', 1),
                          ('l1', 'l1', 'll_11_p', 'll_11_p', 'l+', 1),
                          ('l1', 'l2', 'll_12_p', 'll_12_p', 'l+', 1),
                          ('l1', 'l3', 'll_13_p', 'll_13_p', 'l+', 1),
                          ('l1', 'l4', 'll_14_p', 'll_14_p', 'l+', 1),
                          ('l2', 'l2', 'll_22_p', 'll_22_p', 'l+', 1),
                          ('l2', 'l3', 'll_23_p', 'll_23_p', 'l+', 1),
                          ('l2', 'l4', 'll_24_p', 'll_24_p', 'l+', 1),
                          ('l3', 'l3', 'll_33_p', 'll_33_p', 'l+', 1),
                          ('l3', 'l4', 'll_34_p', 'll_34_p', 'l+', 1),
                          ('l4', 'l4', 'll_44_p', 'll_44_p', 'l+', 1),
                          ('l1', 'l1', 'll_11_m', 'll_11_m', 'l-', 1),
                          ('l1', 'l2', 'll_12_m', 'll_12_m', 'l-', 1),
                          ('l1', 'l3', 'll_13_m', 'll_13_m', 'l-', 1),
                          ('l1', 'l4', 'll_14_m', 'll_14_m', 'l-', 1),
                          ('l2', 'l2', 'll_22_m', 'll_22_m', 'l-', 1),
                          ('l2', 'l3', 'll_23_m', 'll_23_m', 'l-', 1),
                          ('l2', 'l4', 'll_24_m', 'll_24_m', 'l-', 1),
                          ('l3', 'l3', 'll_33_m', 'll_33_m', 'l-', 1),
                          ('l3', 'l4', 'll_34_m', 'll_34_m', 'l-', 1),
                          ('l4', 'l4', 'll_44_m', 'll_44_m', 'l-', 1)])
def test_xi(set_up, corr_method, t1, t2, bm, er, kind, pref):
    cosmo, trcs, bms, ers, fls = set_up
    method, errfac = corr_method
    cl = ccl.angular_cl(cosmo, trcs[t1], trcs[t2], fls['ells'])
    ell = np.arange(fls['lmax'])
    cli = interp1d(fls['ells'], cl, kind='cubic')(ell)
    xi = ccl.correlation(cosmo, ell, cli, bms['theta'],
                         corr_type=kind, method=method)
    xi *= pref
    assert np.all(np.fabs(xi - bms[bm]) < ers[er] * errfac)
