import numpy as np
import pytest
import itertools
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                 matter_power_spectrum='linear')
COSMO.compute_nonlin_power()
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
HMC = ccl.halos.HMCalculator(
    mass_function=HMF, halo_bias=HBF, mass_def=M200, nM=2)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)

NFW = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON,
                               fourier_analytic=True)
HOD = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON)
HOD_nogc = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON)
HOD_nogc.is_number_counts = False
GNFW = ccl.halos.HaloProfilePressureGNFW(mass_def=M200)

PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
AA = np.linspace(0.3, 1.0, 4)


@pytest.mark.parametrize(
    "p1,p2,cv12,p3,p4,cv34,pk",
    [(NFW, None, None, None, None, None, None),
     (HOD, None, None, None, None, None, None),
     (NFW, HOD, None, None, None, None, None),
     (NFW, None, None, GNFW, None, None, None),
     (NFW, None, None, None, NFW, None, None),
     (NFW, HOD, None, GNFW, NFW, None, None),
     (HOD, HOD, PKCH, HOD, HOD, None, None),
     (NFW, HOD, PKC, GNFW, NFW, PKC, None),
     (NFW, None, None, None, None, None, "linear"),
     (NFW, None, None, None, None, None, "nonlinear"),
     (NFW, None, None, None, None, None, COSMO.get_nonlin_power())])
def test_tkkssc_smoke(p1, p2, cv12, p3, p4, cv34, pk):
    tkk = ccl.halos.halomod_Tk3D_SSC(
        COSMO, HMC,
        prof=p1, prof2=p2, prof12_2pt=cv12,
        prof3=p3, prof4=p4, prof34_2pt=cv34,
        p_of_k_a=pk, lk_arr=np.log(KK), a_arr=AA)

    assert (np.isfinite(tkk(0.1, 0.5))).all()


def test_tkkssc_linear_bias_smoke():
    tkkl = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, HMC, prof=NFW, p_of_k_a="linear")
    *_, tkkl_arrs = tkkl.get_spline_arrays()
    assert all([(np.isfinite(tk)).all() for tk in tkkl_arrs])

    tkknl = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, HMC, prof=NFW, p_of_k_a="nonlinear")
    *_, tkknl_arrs = tkknl.get_spline_arrays()
    assert all([(np.isfinite(tk)).all() for tk in tkknl_arrs])

    tkk_pk = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, HMC, prof=NFW, p_of_k_a=COSMO.get_nonlin_power())
    *_, tkk_pk_arrs = tkk_pk.get_spline_arrays()
    assert all([(np.isfinite(tk)).all() for tk in tkk_pk_arrs])


@pytest.mark.parametrize(
    "isNC1,isNC2,isNC3,isNC4",
    itertools.product([True, False], repeat=4))
def test_tkkssc_linear_bias(isNC1, isNC2, isNC3, isNC4):
    bias1, bias2, bias3, bias4 = 2, 3, 4, 5

    # Test that if we remove the biases 12 will be the same as 34.
    tkkl = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, HMC, prof=NFW,
        bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,
        is_number_counts1=False, is_number_counts2=False,
        is_number_counts3=False, is_number_counts4=False,
        lk_arr=np.log(KK), a_arr=AA)

    *_, (tkkl_12, tkkl_34) = tkkl.get_spline_arrays()
    tkkl_12 /= (bias1 * bias2)
    tkkl_34 /= (bias3 * bias4)
    assert np.allclose(tkkl_12, tkkl_34, atol=0, rtol=1e-12)

    # Test with the full T(k1,k2,a) for an NFW profile with bias ~1.
    tkk = ccl.halos.halomod_Tk3D_SSC(
        COSMO, HMC, prof=NFW, lk_arr=np.log(KK), a_arr=AA)
    *_, (tkk_12, tkk_34) = tkk.get_spline_arrays()
    assert np.allclose(tkkl_12, tkk_12, atol=0, rtol=5e-3)
    assert np.allclose(tkkl_34, tkk_34, atol=0, rtol=5e-3)

    # Test with clustering profile.
    tkkl_nc = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, HMC, prof=NFW,
        bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,
        is_number_counts1=isNC1, is_number_counts2=isNC2,
        is_number_counts3=isNC3, is_number_counts4=isNC4,
        lk_arr=np.log(KK), a_arr=AA)
    *_, (tkkl_nc_12, tkkl_nc_34) = tkkl_nc.get_spline_arrays()
    tkkl_nc_12 /= (bias1 * bias2)
    tkkl_nc_34 /= (bias3 * bias4)

    # recover the factors
    factor12 = isNC1*bias1 + isNC2*bias2
    factor34 = isNC3*bias3 + isNC4*bias4

    # calculate what the Tkk's would be with the counterterms.
    tkkl_ct_12 = (tkkl_12 - tkkl_nc_12) / factor12 if factor12 else None
    tkkl_ct_34 = (tkkl_34 - tkkl_nc_34) / factor34 if factor34 else None

    if factor12*factor34:
        assert np.allclose(tkkl_ct_12, tkkl_ct_34, atol=0, rtol=1e-5)


def test_tkkssc_warns():
    """Test that it warns if the profile is negative and use_log is True."""
    Pneg = ccl.halos.HaloProfilePressureGNFW(P0=-1, mass_def=HMC.mass_def)
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.halomod_Tk3D_SSC(
            COSMO, HMC, prof=GNFW, prof2=Pneg,
            lk_arr=np.log(KK), a_arr=AA, use_log=True)


def test_tkkssc_linear_bias_raises():
    """Test that it raises if the profile is not NFW."""
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, HMC, prof=GNFW)


def get_ssc_counterterm_gc(k, a, hmc, prof1, prof2, prof12_2pt):
    P_12 = b1 = b2 = np.zeros_like(k)
    if prof1.is_number_counts or prof2.is_number_counts:
        norm1 = prof1.get_normalization(COSMO, a, hmc=hmc)
        norm2 = prof2.get_normalization(COSMO, a, hmc=hmc)

        i11_1 = hmc.I_1_1(COSMO, k, a, prof1)/norm1
        i11_2 = hmc.I_1_1(COSMO, k, a, prof2)/norm2
        i02_12 = hmc.I_0_2(COSMO, k, a, prof1,
                           prof_2pt=prof12_2pt,
                           prof2=prof2)/(norm1*norm2)

        pk = ccl.linear_matter_power(COSMO, k, a)
        P_12 = pk * i11_1 * i11_2 + i02_12

        if prof1.is_number_counts:
            b1 = ccl.halos.halomod_bias_1pt(COSMO, hmc, k, a, prof1)
        if prof2.is_number_counts:
            b2 = ccl.halos.halomod_bias_1pt(COSMO, hmc, k, a, prof2)

    return (b1 + b2) * P_12


@pytest.mark.parametrize('kwargs', [
                         # All is_number_counts = False
                         {'prof': NFW, 'prof2': NFW,
                          'prof3': NFW, 'prof4': NFW},
                         {'prof': HOD, 'prof2': NFW,
                          'prof3': NFW, 'prof4': NFW},
                         {'prof': HOD, 'prof2': HOD,
                          'prof3': NFW, 'prof4': NFW},
                         {'prof': HOD, 'prof2': HOD,
                          'prof3': HOD, 'prof4': NFW},
                         {'prof': NFW, 'prof2': NFW,
                          'prof3': HOD, 'prof4': HOD},
                         {'prof': HOD, 'prof2': NFW,
                          'prof3': HOD, 'prof4': HOD},
                         {'prof': HOD, 'prof2': HOD,
                          'prof3': NFW, 'prof4': HOD},
                         {'prof': HOD, 'prof2': None,
                          'prof3': NFW, 'prof4': None},
                         {'prof': HOD, 'prof2': None,
                          'prof3': None, 'prof4': None},
                         {'prof': HOD, 'prof2': HOD,
                          'prof3': None, 'prof4': None},
                         # As in benchmarks/test_covariances.py
                         {'prof': NFW, 'prof2': NFW,
                          'prof3': None, 'prof4': None},
                         # Setting prof34_2pt
                         {'prof': NFW, 'prof2': NFW,
                          'prof3': None, 'prof4': None,
                          'prof34_2pt': PKC},
                         # All is_number_counts = True
                         {'prof': HOD, 'prof2': HOD,
                          'prof3': HOD, 'prof4': HOD},
                         ]
                         )
def test_tkkssc_counterterms_gc(kwargs):
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Tk's without clustering terms. Set is_number_counts=False for HOD
    # profiles
    kwargs_nogc = kwargs.copy()
    keys = list(kwargs.keys())
    for k in keys:
        v = kwargs[k]
        if isinstance(v, ccl.halos.HaloProfileHOD):
            kwargs_nogc[k] = HOD_nogc
    tkk_nogc = ccl.halos.halomod_Tk3D_SSC(COSMO, HMC,
                                          lk_arr=np.log(k_arr), a_arr=a_arr,
                                          **kwargs_nogc)
    _, _, _, tkk_nogc_arrs = tkk_nogc.get_spline_arrays()
    tk_nogc_12, tk_nogc_34 = tkk_nogc_arrs

    # Tk's with clustering terms
    tkk_gc = ccl.halos.halomod_Tk3D_SSC(COSMO, HMC,
                                        lk_arr=np.log(k_arr), a_arr=a_arr,
                                        **kwargs)
    _, _, _, tkk_gc_arrs = tkk_gc.get_spline_arrays()
    tk_gc_12, tk_gc_34 = tkk_gc_arrs

    # Update the None's to their corresponding values
    if kwargs['prof2'] is None:
        kwargs['prof2'] = kwargs['prof']
    if kwargs['prof3'] is None:
        kwargs['prof3'] = kwargs['prof']
    if kwargs['prof4'] is None:
        kwargs['prof4'] = kwargs['prof2']

    # Tk's of the clustering terms
    tkc12 = []
    tkc34 = []
    for aa in a_arr:
        tkc12.append(get_ssc_counterterm_gc(k_arr, aa, HMC, kwargs['prof'],
                                            kwargs['prof2'], PKC))
        tkc34.append(get_ssc_counterterm_gc(k_arr, aa, HMC, kwargs['prof3'],
                                            kwargs['prof4'], PKC))
    tkc12 = np.array(tkc12)
    tkc34 = np.array(tkc34)

    assert np.abs((tk_nogc_12 - tkc12) / tk_gc_12 - 1).max() < 1e-5
    assert np.abs((tk_nogc_34 - tkc34) / tk_gc_34 - 1).max() < 1e-5
