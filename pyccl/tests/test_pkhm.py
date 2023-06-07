import numpy as np
import pytest
import pyccl as ccl
from .test_cclobject import check_eq_repr_hash


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON,
                              fourier_analytic=True)
P2 = P1
P3 = ccl.halos.HaloProfilePressureGNFW(mass_def=M200)
PKC = ccl.halos.Profile2pt()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PK2D = ccl.Pk2D.from_function(lambda k, a: a / k)


def test_prof2pt_smoke():
    uk_NFW = P1.fourier(COSMO, KK, MM, AA)
    uk_EIN = P2.fourier(COSMO, KK, MM, AA)
    # Variance
    cv_NN = PKC.fourier_2pt(COSMO, KK, MM, AA, P1)
    assert np.all(np.fabs((cv_NN - uk_NFW**2)) < 1E-10)

    # 2-point
    cv_NE = PKC.fourier_2pt(COSMO, KK, MM, AA, P1, prof2=P2)
    assert np.all(np.fabs((cv_NE - uk_NFW * uk_EIN)) < 1E-10)


def smoke_assert_pkhm_real(func):
    sizes = [(0, 0),
             (2, 0),
             (0, 2),
             (2, 3),
             (1, 3),
             (3, 1)]
    shapes = [(),
              (2,),
              (2,),
              (2, 3),
              (1, 3),
              (3, 1)]
    for (sa, sk), sh in zip(sizes, shapes):
        if sk == 0:
            k = 0.1
        else:
            k = np.logspace(-2., 0., sk)
        if sa == 0:
            a = 1.
        else:
            a = np.linspace(0.5, 1., sa)
        p = func(k, a)
        assert np.shape(p) == sh
        assert np.all(np.isfinite(p))


def test_HMIngredients_eq_repr_hash():
    # Test eq, repr, hash for the HMCalculator and its ingredients.
    # 1. Build a halo model calculator using the default parametrizations.
    HMC = ccl.halos.HMCalculator(
        mass_function="Tinker08", halo_bias="Tinker10", mass_def="200m")

    # 2. Define separate default halo model ingredients.
    MDEF = ccl.halos.MassDef200m
    HMF = ccl.halos.MassFuncTinker08(mass_def=MDEF)
    HBF = ccl.halos.HaloBiasTinker10(mass_def=MDEF)
    HMC2 = ccl.halos.HMCalculator(
        mass_function=HMF, halo_bias=HBF, mass_def=MDEF)  # equal
    HMC3 = ccl.halos.HMCalculator(
        mass_function="Press74", halo_bias="Sheth01", mass_def="fof")  # not eq

    # 3. Test equivalence.
    assert check_eq_repr_hash(MDEF, HMC.mass_def)
    assert check_eq_repr_hash(HMF, HMC.mass_function)
    assert check_eq_repr_hash(HBF, HMC.halo_bias)
    assert check_eq_repr_hash(HMC, HMC2)

    assert check_eq_repr_hash(MDEF, HMC3.mass_def, equal=False)
    assert check_eq_repr_hash(HMF, HMC3.mass_function, equal=False)
    assert check_eq_repr_hash(HBF, HMC3.halo_bias, equal=False)
    assert check_eq_repr_hash(HMC, HMC3, equal=False)


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_mean_profile_smoke(norm):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nM=2)

    def f(k, a):
        return ccl.halos.halomod_mean_profile_1pt(COSMO, hmc, k, a, P1)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('prof', [P1, P3])
def test_pkhm_bias_smoke(prof):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nM=2)

    def f(k, a):
        return ccl.halos.halomod_bias_1pt(COSMO, hmc, k, a, prof)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize(
    "cv,pk,h1,h2,itg,p2",
    [(None, "linear", True, True, "simpson", None),
     (PKC, "linear", True, True, "simpson", None),
     (None, "linear", True, True, "simpson", P3),
     (None, "nonlinear", True, True, "simpson", None),
     (None, PK2D, True, True, "simpson", None),
     (None, None, True, True, "simpson", None),
     (None, "linear", False, True, "simpson", None),
     (None, "linear", True, False, "simpson", None),
     (None, "linear", False, False, "simpson", None),
     (None, "linear", True, True, "spline", None),
     (None, "linear", True, True, "simpson", P2)])
def test_pkhm_pk_smoke(cv, pk, h1, h2, itg, p2):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nM=2)

    def f(k, a):
        return ccl.halos.halomod_power_spectrum(COSMO, hmc, k, a,
                                                prof=P1, prof2=p2,
                                                prof_2pt=cv, p_of_k_a=pk,
                                                get_1h=h1, get_2h=h2)
    smoke_assert_pkhm_real(f)


def test_pkhm_pk2d():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.linspace(0.3, 1, 10)
    pk_arr = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1)

    # Input sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1,
                                  lk_arr=np.log(k_arr), a_arr=a_arr)
    pk_arr_2 = pk2d(k_arr, a_arr, COSMO)
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Standard sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1)
    pk_arr_2 = pk2d(k_arr, a_arr, COSMO)
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Testing profiles which are not equivalent (but very close)
    G1 = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON,
                                  log10Mmin_0=12.00000)
    G2 = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON,
                                  log10Mmin_0=11.99999)
    assert G1 != G2

    # I_1_1
    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, G1,
                                           prof2=G1)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, G1,
                                           prof2=G2)
    assert np.allclose(pk1, pk0, rtol=1e-4)

    # Profile normalization
    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, G1,
                                           prof2=G1)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, G1,
                                           prof2=G2)
    assert np.allclose(pk1, pk0, rtol=1e-4)

    # I_0_2 & I_1_2
    assert np.allclose(hmc.I_0_2(COSMO, KK, AA, P1, prof_2pt=PKC),
                       hmc.I_0_2(COSMO, KK, AA, P1, prof2=P1, prof_2pt=PKC),
                       rtol=0)
    assert np.allclose(hmc.I_1_2(COSMO, KK, AA, P1, prof_2pt=PKC),
                       hmc.I_1_2(COSMO, KK, AA, P1, prof2=P1, prof_2pt=PKC),
                       rtol=0)
    # I_0_22
    I0 = hmc.I_0_22(COSMO, KK, AA, P1, prof2=P1, prof3=P1, prof4=P1,
                    prof12_2pt=PKC, prof34_2pt=PKC)
    assert np.allclose(hmc.I_0_22(COSMO, KK, AA,
                                  P1, prof2=P1, prof3=P1, prof4=P1,
                                  prof12_2pt=PKC, prof34_2pt=None),
                       I0, rtol=0)
    assert np.allclose(hmc.I_0_22(COSMO, KK, AA,
                                  P1, prof2=P1, prof3=None, prof4=None,
                                  prof12_2pt=PKC, prof34_2pt=PKC),
                       I0, rtol=0)

    # 1h/2h transition
    def alpha0(a):  # no smoothing
        return 1.

    def alpha1(a):
        return 0.7

    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           smooth_transition=None)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           smooth_transition=alpha0)
    assert np.allclose(pk0, pk1, rtol=0)
    pk2 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           smooth_transition=alpha1)
    assert np.all(pk2/pk0 > 1.)

    # 1-halo damping
    def ks0(a):  # no damping
        return 1e-16

    def ks1(a):  # fully suppressed
        return 1e16

    def ks2(a):  # reasonable
        return 0.04

    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           suppress_1h=None, get_2h=False)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           suppress_1h=ks0, get_2h=False)
    assert np.allclose(pk0, pk1, rtol=0)
    pk2 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           suppress_1h=ks1, get_2h=False)
    assert np.allclose(pk2, 0, rtol=0)
    pk3 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           suppress_1h=ks2, get_2h=False)
    fact = (k_arr/0.04)**4 / (1 + (k_arr/0.04)**4)
    assert np.allclose(pk3, pk0*fact, rtol=0)


def test_pkhm_errors():
    # Wrong integration
    with pytest.raises(ValueError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                               mass_def=M200, integration_method_M='Sampson')

    # Inconsistent mass definitions
    m200c = ccl.halos.MassDef.create_instance("200c")
    m200m = ccl.halos.MassDef.create_instance("200m")
    hmf = ccl.halos.MassFunc.create_instance("Tinker08", mass_def=m200c)
    hbf = ccl.halos.HaloBias.create_instance("Tinker10", mass_def=m200m)
    with pytest.raises(ValueError):
        ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                               mass_def=m200c)

    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)

    # Wrong pk2d
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         p_of_k_a=KK)

    def func():
        pass

    # Wrong 1h/2h smoothing
    with pytest.raises(ValueError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         smooth_transition=func, get_1h=False)

    # Wrong 1h damping
    with pytest.raises(ValueError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         suppress_1h=func, get_1h=False)


def test_hmcalculator_from_string_smoke():
    hmc1 = ccl.halos.HMCalculator(
        mass_function=HMF, halo_bias=HBF, mass_def=M200)
    hmc2 = ccl.halos.HMCalculator(
        mass_function="Tinker10", halo_bias="Tinker10", mass_def="200m")
    for attr in ["mass_function", "halo_bias", "mass_def"]:
        assert getattr(hmc1, attr).name == getattr(hmc2, attr).name


def test_hmcalculator_from_string_raises():
    # Check that if the necessary arguments aren't provided, it raises.
    kw = {"mass_function": "Tinker10", "halo_bias": "Tinker10"}
    with pytest.raises(ValueError):
        # needs a mass_def to pass to the halo model ingredients
        ccl.halos.HMCalculator(**kw)
    # but this one is fine
    ccl.halos.HMCalculator(**kw, mass_def="200c")


def test_hmcalculator_inconsistent_mass_def_raises():
    # Check that HMCalculator complains for inconsistent mass definitions.
    hmc = ccl.halos.HMCalculator(
        mass_function="Tinker10", halo_bias="Tinker10", mass_def="200c")
    prof = ccl.halos.HaloProfilePressureGNFW(mass_def="500c")
    with pytest.raises(ValueError):
        hmc._check_mass_def(prof)
