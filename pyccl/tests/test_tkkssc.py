import numpy as np
import pytest
import itertools
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                 matter_power_spectrum='linear')
COSMO.compute_nonlin_power()
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
HMC = ccl.halos.HMCalculator(
    mass_function=HMF, halo_bias=HBF, mass_def=M200, nM=2)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)

NFW = ccl.halos.HaloProfileNFW(concentration=CON, fourier_analytic=True)
HOD = ccl.halos.HaloProfileHOD(concentration=CON)
GNFW = ccl.halos.HaloProfilePressureGNFW()

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
    with pytest.warns(ccl.CCLDeprecationWarning):  # TODO: remove normprof v3
        tkk = ccl.halos.halomod_Tk3D_SSC(
            COSMO, HMC, prof=NFW, lk_arr=np.log(KK), a_arr=AA,
            normprof1=True)
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
    Pneg = ccl.halos.HaloProfilePressureGNFW(P0=-1)
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.halomod_Tk3D_SSC(
            COSMO, HMC, prof=GNFW, prof2=Pneg,
            lk_arr=np.log(KK), a_arr=AA, use_log=True)


def test_tkkssc_linear_bias_raises():
    """Test that it raises if the profile is not NFW."""
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC_linear_bias(COSMO, HMC, prof=GNFW)
