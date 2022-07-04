import itertools
import numpy as np
import pytest
import pyccl as ccl
from pyccl.halos.halo_model import halomod_bias_1pt
from pyccl.pyutils import assert_warns


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
COSMO.compute_nonlin_power()
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(COSMO, mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(COSMO, mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200),
                              fourier_analytic=True)
# P2 will have is_number_counts = True
P2 = ccl.halos.HaloProfileHOD(ccl.halos.ConcentrationDuffy08(M200))
P2_nogc = ccl.halos.HaloProfileHOD(ccl.halos.ConcentrationDuffy08(M200))
P2_nogc.is_number_counts = False
P3 = ccl.halos.HaloProfilePressureGNFW()
P4 = P1
Pneg = ccl.halos.HaloProfilePressureGNFW(P0=-1)
PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0


def get_ssc_counterterm_gc(k, a, hmc, prof1, prof2, prof12_2pt,
                           normalize=False):

    P_12 = b1 = b2 = np.zeros_like(k)
    if prof1.is_number_counts or prof2.is_number_counts:
        norm1 = hmc.profile_norm(COSMO, a, prof1)
        norm2 = hmc.profile_norm(COSMO, a, prof2)
        norm12 = 1
        if prof1.is_number_counts or normalize:
            norm12 *= norm1
        if prof2.is_number_counts or normalize:
            norm12 *= norm2

        i11_1 = hmc.I_1_1(COSMO, k, a, prof1)
        i11_2 = hmc.I_1_1(COSMO, k, a, prof2)
        i02_12 = hmc.I_0_2(COSMO, k, a, prof1, prof12_2pt, prof2)

        pk = ccl.linear_matter_power(COSMO, k, a)
        P_12 = norm12 * (pk * i11_1 * i11_2 + i02_12)

        if prof1.is_number_counts:
            b1 = halomod_bias_1pt(COSMO, hmc, k, a, prof1) * norm1
        if prof2.is_number_counts:
            b2 = halomod_bias_1pt(COSMO, hmc, k, a, prof2) * norm2

    return (b1 + b2) * P_12


@pytest.mark.parametrize('pars',
                         [{'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': P3, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': P4, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': P3, 'p4': P4, 'cv34': None,
                           'norm': True, 'pk': None},
                          {'p1': P2, 'p2': P2, 'cv12': PKCH,
                           'p3': P2, 'p4': P2, 'cv34': None,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': PKC,
                           'p3': P3, 'p4': P4, 'cv34': PKC,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': 'linear'},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': 'nonlinear'},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': COSMO.get_nonlin_power()},
                          ],)
def test_tkkssc_smoke(pars):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    tkk = ccl.halos.halomod_Tk3D_SSC(COSMO, hmc,
                                     prof1=pars['p1'],
                                     prof2=pars['p2'],
                                     prof12_2pt=pars['cv12'],
                                     prof3=pars['p3'],
                                     prof4=pars['p4'],
                                     prof34_2pt=pars['cv34'],
                                     normprof1=pars['norm'],
                                     normprof2=pars['norm'],
                                     normprof3=pars['norm'],
                                     normprof4=pars['norm'],
                                     p_of_k_a=pars['pk'],
                                     lk_arr=np.log(k_arr), a_arr=a_arr,
                                     )
    tk = tkk.eval(0.1, 0.5)
    assert np.all(np.isfinite(tk))


def test_tkkssc_errors():

    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Wrong first profile
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, None)
    # Wrong other profiles
    for i in range(2, 4):
        kw = {'prof%d' % i: PKC}
        with pytest.raises(TypeError):
            ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1, **kw)
    # Wrong 2pts
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1,
                                   prof12_2pt=P2)
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1,
                                   prof34_2pt=P2)

    # No normalization for number counts profile
    with pytest.raises(ValueError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P2, normprof1=False)
    with pytest.raises(ValueError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1, prof2=P2, normprof2=False)
    with pytest.raises(ValueError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1, prof3=P2, normprof3=False)
    with pytest.raises(ValueError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1, prof4=P2, normprof4=False)

    # Negative profile in logspace
    assert_warns(ccl.CCLWarning, ccl.halos.halomod_Tk3D_1h,
                 COSMO, hmc, P3, prof2=Pneg,
                 lk_arr=np.log(k_arr), a_arr=a_arr,
                 use_log=True)


@pytest.mark.parametrize('kwargs', [
                         # All is_number_counts = False
                         {'prof1': P1, 'prof2': P1,
                          'prof3': P1, 'prof4': P1},
                         #
                         {'prof1': P2, 'prof2': P1,
                          'prof3': P1, 'prof4': P1},
                         #
                         {'prof1': P2, 'prof2': P2,
                          'prof3': P1, 'prof4': P1},
                         #
                         {'prof1': P2, 'prof2': P2,
                          'prof3': P2, 'prof4': P1},
                         #
                         {'prof1': P1, 'prof2': P1,
                          'prof3': P2, 'prof4': P2},
                         #
                         {'prof1': P2, 'prof2': P1,
                          'prof3': P2, 'prof4': P2},
                         #
                         {'prof1': P2, 'prof2': P2,
                          'prof3': P1, 'prof4': P2},
                         #
                         {'prof1': P2, 'prof2': None,
                          'prof3': P1, 'prof4': None},
                         #
                         {'prof1': P2, 'prof2': None,
                          'prof3': None, 'prof4': None},
                         #
                         {'prof1': P2, 'prof2': P2,
                          'prof3': None, 'prof4': None},
                         # As in benchmarks/test_covariances.py
                         {'prof1': P1, 'prof2': P1,
                          'prof3': None, 'prof4': None},
                         # Setting prof34_2pt
                         {'prof1': P1, 'prof2': P1,
                          'prof3': None, 'prof4': None,
                          'prof34_2pt': PKC},
                         # All is_number_counts = True
                         {'prof1': P2, 'prof2': P2,
                          'prof3': P2, 'prof4': P2},

                         ]
                         )
def test_tkkssc_counterterms_gc(kwargs):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Tk's without clustering terms. Set is_number_counts=False for HOD
    # profiles
    # Ensure HOD profiles are normalized
    kwargs_nogc = kwargs.copy()
    keys = list(kwargs.keys())
    for k in keys:
        v = kwargs[k]
        if isinstance(v, ccl.halos.HaloProfileHOD):
            kwargs_nogc[k] = P2_nogc
            kwargs_nogc['norm' + k] = True
            kwargs['norm' + k] = True
    tkk_nogc = ccl.halos.halomod_Tk3D_SSC(COSMO, hmc,
                                          lk_arr=np.log(k_arr), a_arr=a_arr,
                                          **kwargs_nogc)
    _, _, _, tkk_nogc_arrs = tkk_nogc.get_spline_arrays()
    tk_nogc_12, tk_nogc_34 = tkk_nogc_arrs

    # Tk's with clustering terms
    tkk_gc = ccl.halos.halomod_Tk3D_SSC(COSMO, hmc,
                                        lk_arr=np.log(k_arr), a_arr=a_arr,
                                        **kwargs)
    _, _, _, tkk_gc_arrs = tkk_gc.get_spline_arrays()
    tk_gc_12, tk_gc_34 = tkk_gc_arrs

    # Update the None's to their corresponding values
    if kwargs['prof2'] is None:
        kwargs['prof2'] = kwargs['prof1']
    if kwargs['prof3'] is None:
        kwargs['prof3'] = kwargs['prof1']
    if kwargs['prof4'] is None:
        kwargs['prof4'] = kwargs['prof3']

    # Tk's of the clustering terms
    tkc12 = []
    tkc34 = []
    for aa in a_arr:
        tkc12.append(get_ssc_counterterm_gc(k_arr, aa, hmc, kwargs['prof1'],
                                            kwargs['prof2'], PKC))
        tkc34.append(get_ssc_counterterm_gc(k_arr, aa, hmc, kwargs['prof3'],
                                            kwargs['prof4'], PKC))
    tkc12 = np.array(tkc12)
    tkc34 = np.array(tkc34)

    assert np.abs((tk_nogc_12 - tkc12) / tk_gc_12 - 1).max() < 1e-5
    assert np.abs((tk_nogc_34 - tkc34) / tk_gc_34 - 1).max() < 1e-5


@pytest.mark.parametrize('kwargs', [{f'is_number_counts{i+1}': nc[i] for i in
                                     range(4)} for nc in
                                    itertools.product([True, False],
                                                      repeat=4)])
def test_tkkssc_linear_bias(kwargs):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Tk's exact version
    prof = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200),
                                    fourier_analytic=True)
    bias1 = 2
    bias2 = 3
    bias3 = 4
    bias4 = 5
    is_nc = False

    # Tk's from tkkssc_linear
    tkk_lin = ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof,
                                                     bias1=bias1,
                                                     bias2=bias2,
                                                     bias3=bias3,
                                                     bias4=bias4,
                                                     is_number_counts1=is_nc,
                                                     is_number_counts2=is_nc,
                                                     is_number_counts3=is_nc,
                                                     is_number_counts4=is_nc,
                                                     lk_arr=np.log(k_arr),
                                                     a_arr=a_arr)
    _, _, _, tkk_lin_arrs = tkk_lin.get_spline_arrays()
    tk_lin_12, tk_lin_34 = tkk_lin_arrs
    # Remove the biases
    tk_lin_12 /= (bias1 * bias2)
    tk_lin_34 /= (bias3 * bias4)
    assert np.abs(tk_lin_12 / tk_lin_34 - 1).max() < 1e-5

    # True Tk's (biases for NFW ~ 1)
    tkk = ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, prof1=prof,
                                     lk_arr=np.log(k_arr), a_arr=a_arr,
                                     normprof1=True, normprof2=True,
                                     normprof3=True, normprof4=True)
    _, _, _, tkk_arrs = tkk.get_spline_arrays()
    tk_12, tk_34 = tkk_arrs

    assert np.abs(tk_lin_12 / tk_12 - 1).max() < 1e-2
    assert np.abs(tk_lin_34 / tk_34 - 1).max() < 1e-2

    # Now with clustering
    tkk_lin_nc = ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof,
                                                        bias1=bias1,
                                                        bias2=bias2,
                                                        bias3=bias3,
                                                        bias4=bias4,
                                                        **kwargs,
                                                        lk_arr=np.log(k_arr),
                                                        a_arr=a_arr)
    _, _, _, tkk_lin_nc_arrs = tkk_lin_nc.get_spline_arrays()
    tk_lin_nc_12, tk_lin_nc_34 = tkk_lin_nc_arrs
    tk_lin_nc_12 /= (bias1 * bias2)
    tk_lin_nc_34 /= (bias3 * bias4)

    factor12 = 0
    factor12 += bias1 if kwargs['is_number_counts1'] else 0
    factor12 += bias2 if kwargs['is_number_counts2'] else 0

    factor34 = 0
    factor34 += bias3 if kwargs['is_number_counts3'] else 0
    factor34 += bias4 if kwargs['is_number_counts4'] else 0

    tk_lin_ct_12 = np.zeros_like(tk_lin_nc_12)
    if factor12 != 0:
        tk_lin_ct_12 = (tk_lin_12 - tk_lin_nc_12) / factor12  # = pk+i02

    tk_lin_ct_34 = np.zeros_like(tk_lin_nc_34)
    if factor34 != 0:
        tk_lin_ct_34 = (tk_lin_34 - tk_lin_nc_34) / factor34  # = pk+i02

    if (factor12 != 0) and (factor34 != 0):
        assert np.abs(tk_lin_ct_12 / tk_lin_ct_34 - 1).max() < 1e-5

    # True counter terms
    tkc12 = []
    tkc34 = []
    prof.is_number_counts = True  # Trick the function below
    for aa in a_arr:
        # Divide by 2 to account for ~(1 + 1)
        tkc_ia = get_ssc_counterterm_gc(k_arr, aa, hmc, prof, prof, PKC,
                                        normalize=True) / 2
        if factor12 == 0:
            tkc12.append(np.zeros_like(tkc_ia))
        else:
            tkc12.append(tkc_ia)

        if factor34 == 0:
            tkc34.append(np.zeros_like(tkc_ia))
        else:
            tkc34.append(tkc_ia)

    tkc12 = np.array(tkc12)
    tkc34 = np.array(tkc34)

    # Add 1e-100 for the cases when the counter terms are 0
    assert np.abs((tk_lin_ct_12 + 1e-100) / (tkc12 + 1e-100) - 1).max() < 1e-2
    assert np.abs((tk_lin_ct_34 + 1e-100) / (tkc34 + 1e-100) - 1).max() < 1e-2


def test_tkkssc_linear_bias_smoke_and_errors():
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Tk's exact version
    prof = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200),
                                    fourier_analytic=True)

    ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof,
                                           p_of_k_a='linear')

    ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof,
                                           p_of_k_a='nonlinear')

    pk = COSMO.get_nonlin_power()
    ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof, p_of_k_a=pk)

    # Error when prof is not NFW
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=P2)

    # Error when p_of_k_a is wrong
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, hmc, prof=prof,
                                               p_of_k_a=P1)

    # Negative profile in logspace
    assert_warns(ccl.CCLWarning, ccl.halos.halomod_Tk3D_SSC_linear_bias,
                 COSMO, hmc, prof, bias1=-1,
                 lk_arr=np.log(k_arr), a_arr=a_arr,
                 use_log=True)
