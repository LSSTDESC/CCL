import numpy as np
import pytest
from scipy.integrate import simpson
import pyccl as ccl
from .test_cclobject import check_eq_repr_hash


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                 matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker08(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
HMC = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                             mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
NFW = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON)
HOD = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON)
RHO0 = ccl.physical_constants.RHO_CRITICAL * COSMO['Omega_m'] * COSMO['h']**2


def SEL_FUNC(M, a):
    return (M > 1E14) * 1.


def SEL_FUNC_LOW(M, a):
    return (M > 1E13) * (M < 1E14) * 1.


SEL = ccl.halos.HaloProfileSelection(mass_def=M200, selection=SEL_FUNC)
SEL_OFF = ccl.halos.HaloProfileSelection(mass_def=M200, selection=SEL_FUNC,
                                         f_off=0.3, r_off=0.4)
SEL_LOW = ccl.halos.HaloProfileSelection(mass_def=M200,
                                         selection=SEL_FUNC_LOW)
SEL_MASS = ccl.halos.HaloProfileSelection(
    mass_def=M200, selection=lambda M, a: M / RHO0)
KK = np.geomspace(1E-3, 10, 32)
AA = 0.7


def _mass_integrals(a):
    M = HMC._mass
    return (M, HMC._lmass, HMF(COSMO, M, a), HBF(COSMO, M, a),
            SEL_FUNC(M, a))


def _offset(k, M, a):
    R = M200.get_radius(COSMO, M, a) / a
    return 0.7 + 0.3 * np.exp(-0.5 * (k[None, :] * 0.4 * R[:, None])**2)


def test_selection_profile():
    M = np.geomspace(1E12, 1E15, 8)
    uk = SEL.fourier(COSMO, KK, M, AA)
    assert uk.shape == (8, 32)
    assert np.all(uk == SEL_FUNC(M, AA)[:, None])
    assert np.shape(SEL.fourier(COSMO, 0.1, 1E15, AA)) == ()

    uk = SEL_OFF.fourier(COSMO, KK, M, AA)
    assert np.allclose(uk, SEL_FUNC(M, AA)[:, None] * _offset(KK, M, AA),
                       rtol=1E-12)
    assert np.allclose(SEL_OFF.fourier(COSMO, 1E-5, M, AA), SEL_FUNC(M, AA))

    M, lM, n, _, S = _mass_integrals(AA)
    norm = SEL.get_normalization(COSMO, AA, hmc=HMC)
    assert np.isclose(norm, simpson(n * S, x=lM), rtol=1E-10)

    sel = ccl.halos.HaloProfileSelection(mass_def=M200, selection=SEL_FUNC)
    assert check_eq_repr_hash(SEL, sel)
    assert check_eq_repr_hash(SEL, SEL_OFF, equal=False)


def test_profile2pt_selection():
    M = np.geomspace(1E12, 1E15, 8)
    S = SEL_FUNC(M, AA)
    p2pt = ccl.halos.Profile2ptSelection()

    for p in [SEL, SEL_OFF]:
        uk = p2pt.fourier_2pt(COSMO, KK, M, AA, p)
        assert np.all(uk == S[:, None])

    uk = p2pt.fourier_2pt(COSMO, KK, M, AA, SEL, prof2=SEL_OFF)
    assert np.allclose(uk, S[:, None] * _offset(KK, M, AA), rtol=1E-12)
    uk = p2pt.fourier_2pt(COSMO, KK, M, AA, SEL, prof2=SEL_OFF, diag=False)
    assert uk.shape == (8, 32, 32)
    assert np.allclose(uk[:, 3, :], S[:, None] * _offset(KK, M, AA)[:, [3]])

    with pytest.raises(ValueError):
        p2pt.fourier_2pt(COSMO, KK, M, AA, SEL, prof2=SEL_LOW)
    p2pt = ccl.halos.Profile2ptSelection(
        joint_selection=lambda M, a: 0.5 * M / M)
    uk = p2pt.fourier_2pt(COSMO, KK, M, AA, SEL, prof2=SEL_LOW)
    assert np.all(uk == 0.5)
    assert np.shape(p2pt.fourier_2pt(COSMO, 0.1, 1E14, AA, SEL,
                                     prof2=SEL_LOW)) == ()

    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, KK, M, AA, SEL, prof2=NFW)


def test_I_0_3():
    M, lM, n, _, S = _mass_integrals(AA)
    uk = NFW.fourier(COSMO, KK, M, AA)
    i03 = HMC.I_0_3(COSMO, KK, AA, SEL, prof2=NFW,
                    prof_2pt=ccl.halos.Profile2pt())
    ref = simpson(n[:, None] * S[:, None] * uk**2, x=lM, axis=0)
    assert np.allclose(i03, ref, rtol=1E-10)


def test_bispectrum_takada_bridle():
    M, lM, n, b, S = _mass_integrals(AA)
    u = NFW.fourier(COSMO, KK, M, AA) / RHO0
    pk = COSMO.linear_matter_power(KK, AA)
    b1h = simpson(n[:, None] * S[:, None] * u**2, x=lM, axis=0)
    i11 = HMC.I_1_1(COSMO, KK, AA, NFW) / RHO0
    b2h = 2 * i11 * pk * simpson(n[:, None] * b[:, None] * S[:, None] * u,
                                 x=lM, axis=0)

    B = ccl.halos.halomod_bispectrum_squeezed
    bk_1h = B(COSMO, HMC, KK, AA, SEL, prof2=NFW, get_2h=False)
    bk_2h = B(COSMO, HMC, KK, AA, SEL, prof2=NFW, get_1h=False)
    bk = B(COSMO, HMC, KK, AA, SEL, prof2=NFW)
    assert np.allclose(bk_1h, b1h, rtol=1E-10)
    assert np.allclose(bk_2h, b2h, rtol=1E-10)
    assert np.allclose(bk, b1h + b2h, rtol=1E-10)

    bk_n = B(COSMO, HMC, KK, AA, SEL, prof2=NFW, normalize_counts=True)
    norm = SEL.get_normalization(COSMO, AA, hmc=HMC)
    assert np.allclose(bk_n, bk / norm, rtol=1E-10)


def test_bispectrum_halo_halo_matter():
    M, lM, n, b, S = _mass_integrals(AA)
    u = NFW.fourier(COSMO, KK, M, AA) / RHO0
    nbar = simpson(n * S, x=lM)
    B = ccl.halos.halomod_bispectrum_squeezed

    bk = B(COSMO, HMC, KK, AA, SEL, prof2=SEL, prof3=NFW, get_2h=False)
    ref = simpson(n[:, None] * S[:, None] * u, x=lM, axis=0) / nbar
    assert np.allclose(bk, ref, rtol=1E-10)
    bk2 = B(COSMO, HMC, KK, AA, SEL, prof2=NFW, prof3=SEL, get_2h=False)
    assert np.allclose(bk, bk2, rtol=1E-12)

    off = _offset(KK, M, AA)
    ref = simpson(n[:, None] * S[:, None] * off * u, x=lM, axis=0) / nbar
    for p in [SEL, SEL_OFF]:
        bk = B(COSMO, HMC, KK, AA, p, prof2=SEL_OFF, prof3=NFW,
               get_2h=False)
        assert np.allclose(bk, ref, rtol=1E-10)

    p2pt = ccl.halos.Profile2ptSelection(joint_selection=lambda M, a: 0 * M)
    bk = B(COSMO, HMC, KK, AA, SEL, prof2=SEL_LOW, prof3=NFW,
           prof12_2pt=p2pt, get_2h=False)
    assert np.all(bk == 0)
    with pytest.raises(ValueError):
        B(COSMO, HMC, KK, AA, SEL, prof2=SEL_LOW, prof3=NFW)


def test_bispectrum_ssc():
    a_arr = np.linspace(0.4, 1., 7)
    tk = ccl.halos.halomod_Tk3D_SSC(COSMO, HMC, NFW, lk_arr=np.log(KK),
                                    a_arr=a_arr)
    _, _, _, (resp, _) = tk.get_spline_arrays()
    bk = ccl.halos.halomod_bispectrum_squeezed_SSC(COSMO, HMC, KK, a_arr,
                                                   SEL_MASS, prof2=NFW)
    assert np.allclose(bk, resp, rtol=1E-10)

    tk = ccl.halos.halomod_Tk3D_SSC(COSMO, HMC, HOD,
                                    prof12_2pt=ccl.halos.Profile2ptHOD(),
                                    lk_arr=np.log(KK), a_arr=a_arr)
    _, _, _, (resp, _) = tk.get_spline_arrays()
    bk = ccl.halos.halomod_bispectrum_squeezed_SSC(
        COSMO, HMC, KK, a_arr, SEL_MASS, prof2=HOD,
        prof23_2pt=ccl.halos.Profile2ptHOD())
    assert np.allclose(bk, resp, rtol=1E-10)

    bk = ccl.halos.halomod_bispectrum_squeezed_SSC(COSMO, HMC, KK, AA, SEL,
                                                   prof2=NFW)
    bk_off = ccl.halos.halomod_bispectrum_squeezed_SSC(COSMO, HMC, KK, AA,
                                                       SEL_OFF, prof2=NFW)
    assert np.allclose(bk, bk_off, rtol=1E-12)


@pytest.mark.parametrize('func', [ccl.halos.halomod_bispectrum_squeezed,
                                  ccl.halos.halomod_bispectrum_squeezed_SSC])
def test_bispectrum_smoke(func):
    for k, a, shape in [(0.1, 1., ()),
                        (KK, 1., (32,)),
                        (0.1, [0.5, 1.], (2,)),
                        (KK, [0.5, 1.], (2, 32))]:
        bk = func(COSMO, HMC, k, a, SEL, prof2=HOD)
        assert np.shape(bk) == shape
        assert np.all(np.isfinite(bk))

    with pytest.raises(TypeError):
        func(COSMO, HMC, KK, AA, NFW, prof2=NFW)


@pytest.mark.parametrize('kind', ['', '_SSC'])
def test_bispectrum_pk2d(kind):
    lk_arr = np.log(KK)
    a_arr = np.linspace(0.3, 1., 8)
    bk2d = getattr(ccl.halos, f'halomod_Pk2D_bispectrum_squeezed{kind}')(
        COSMO, HMC, SEL, prof2=NFW, lk_arr=lk_arr, a_arr=a_arr)
    bk = getattr(ccl.halos, f'halomod_bispectrum_squeezed{kind}')(
        COSMO, HMC, KK, a_arr, SEL, prof2=NFW)
    assert np.allclose(bk2d(KK, a_arr[3]), bk[3], rtol=1E-6)


def test_cov_counts():
    fsky = 0.1
    z = np.linspace(0., 1.5, 256)
    nz = np.exp(-0.5 * ((z - 0.6) / 0.15)**2)
    tr = ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=(z, nz),
                                bias=(z, np.ones_like(z)))
    a_arr = np.linspace(0.3, 1., 32)
    bk = ccl.halos.halomod_Pk2D_bispectrum_squeezed(
        COSMO, HMC, SEL, prof2=NFW, a_arr=a_arr)
    bk_ssc = ccl.halos.halomod_Pk2D_bispectrum_squeezed_SSC(
        COSMO, HMC, SEL, prof2=NFW, a_arr=a_arr)
    ell = np.array([50., 300.])
    cov = ccl.angular_cl_cov_counts(COSMO, tr, tr, ell=ell, b_of_k_a=bk,
                                    fsky=fsky)
    cov_ssc = ccl.angular_cl_cov_counts_SSC(COSMO, tr, tr, ell=ell,
                                            b_of_k_a=bk_ssc, fsky=fsky)


    chi = np.linspace(500., 3500., 1024)
    a = ccl.scale_factor_of_chi(COSMO, chi)
    W = tr.get_kernel(chi)[0]
    s2b = ccl.sigma2_B_disc(COSMO, a_arr=a, fsky=fsky)
    for i, l in enumerate(ell):
        k = (l + 0.5) / chi
        b = np.array([bk(kk, aa) for kk, aa in zip(k, a)])
        b_ssc = np.array([bk_ssc(kk, aa) for kk, aa in zip(k, a)])
        ref = simpson(W**2 / chi**2 * b, x=chi) / (4 * np.pi * fsky)
        ref_ssc = simpson(W**2 * s2b * b_ssc, x=chi)
        assert np.isclose(cov[i], ref, rtol=1E-3)
        assert np.isclose(cov_ssc[i], ref_ssc, rtol=1E-3)

    s2b = (a_arr, ccl.sigma2_B_disc(COSMO, a_arr=a_arr, fsky=fsky))
    cov_ssc2 = ccl.angular_cl_cov_counts_SSC(COSMO, tr, tr, ell=ell,
                                             b_of_k_a=bk_ssc, sigma2_B=s2b)
    assert np.allclose(cov_ssc2, cov_ssc, rtol=1E-6)

    assert np.shape(ccl.angular_cl_cov_counts(COSMO, tr, tr, ell=100.,
                                              b_of_k_a=bk)) == ()
    for func, b in [(ccl.angular_cl_cov_counts, bk),
                    (ccl.angular_cl_cov_counts_SSC, bk_ssc)]:
        with pytest.raises(ValueError):
            func(COSMO, tr, tr, ell=ell, b_of_k_a=b,
                 integration_method='cag_quad')
