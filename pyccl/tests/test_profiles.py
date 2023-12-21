import numpy as np
import pytest
import pyccl as ccl
from pyccl import UnlockInstance
from .test_cclobject import check_eq_repr_hash


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')


def test_HaloProfile_eq_repr_hash():
    # Test eq, repr, hash for HaloProfile and Profile2pt.
    # 1. HaloProfile
    CM1 = ccl.halos.Concentration.from_name("Duffy08")()
    CM2 = ccl.halos.Concentration.from_name("Duffy08")()

<<<<<<< HEAD
    P1 = ccl.halos.HaloProfileHOD(c_M_relation=CM1)
    P2 = ccl.halos.HaloProfileHOD(c_M_relation=CM2)
    assert check_eq_repr_hash(CM1, CM2)
    assert check_eq_repr_hash(P1, P2)

    P1.update_parameters(lMmin_0=P1.lMmin_0/2)
=======
    P1 = ccl.halos.HaloProfileHOD(mass_def=CM1.mass_def,
                                  concentration=CM1)
    P2 = ccl.halos.HaloProfileHOD(mass_def=CM2.mass_def,
                                  concentration=CM2)
    assert check_eq_repr_hash(CM1, CM2)
    assert check_eq_repr_hash(P1, P2)

    P1.update_parameters(log10Mmin_0=P1.log10Mmin_0/2)
>>>>>>> master
    assert check_eq_repr_hash(P1, P2, equal=False)

    # 2. Profile2pt
    PCOV1 = ccl.halos.Profile2pt(r_corr=1.0)
    PCOV2 = ccl.halos.Profile2pt(r_corr=1.0)
    assert check_eq_repr_hash(PCOV1, PCOV2)

    PCOV2.update_parameters(r_corr=1.5)
    assert check_eq_repr_hash(PCOV1, PCOV2, equal=False)


<<<<<<< HEAD
def one_f(cosmo, M, a=1, mass_def=M200):
    if np.ndim(M) == 0:
        return 1
    else:
        return np.ones(M.size)
=======
def one_f(cosmo, M, a=1):
    return 1 if np.isscalar(M) else np.ones(M.size)
>>>>>>> master


def smoke_assert_prof_real(profile, method='_real'):
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
    for (sm, sr), sh in zip(sizes, shapes):
        if sr == 0:
            r = 0.5
        else:
            # Don't include 0 to avoid 1/0 at origin.
            r = np.linspace(0.001, 1., sr)
        if sm == 0:
            m = 1E12
        else:
            m = np.geomspace(1E10, 1E14, sm)
<<<<<<< HEAD
        p = getattr(profile, method)(COSMO, r, m, 1., mass_def=M200)
        assert np.shape(p) == sh


# TODO: Uncomment once the profiles can be compared.
# def test_profiles_equal():
#     M200m = ccl.halos.MassDef200m()
#     cm = ccl.halos.ConcentrationDuffy08(mass_def=M200m)
#     p1 = ccl.halos.HaloProfileHOD(concentration=cm, lMmin_0=12.)
=======
        p = getattr(profile, method)(COSMO, r, m, 1)
        assert np.shape(p) == sh


def test_profiles_equal():
    cm = ccl.halos.ConcentrationDuffy08(mass_def='200m')
    p1 = ccl.halos.HaloProfileHOD(mass_def='200m',
                                  concentration=cm, log10Mmin_0=12.)
>>>>>>> master

    # different profile types
    p2 = ccl.halos.HaloProfilePressureGNFW(mass_def='200m')
    assert p1 != p2

    # equal profiles
    p2 = p1
    assert p1 == p2

<<<<<<< HEAD
#     # equivalent profiles
#     cm2 = ccl.halos.ConcentrationDuffy08(mass_def=M200m)
#     p2 = ccl.halos.HaloProfileHOD(concentration=cm2, lMmin_0=12.)
#     assert p1 == p2

#     # different parameters
#     p2 = ccl.halos.HaloProfileHOD(concentration=cm, lMmin_0=11.)
#     assert p1 != p2

#     # different mass-concentration
#     cm2 = ccl.halos.ConcentrationConstant()
#     p2 = ccl.halos.HaloProfileHOD(concentration=cm2, lMmin_0=12.)
#     assert p1 != p2

#     # different mass-concentration mass definition
#     M200c = ccl.halos.MassDef200c()
#     cm2 = ccl.halos.ConcentrationDuffy08(mass_def=M200c)
#     p2 = ccl.halos.HaloProfileHOD(concentration=cm2, lMmin_0=12.)
#     assert p1 != p2

#     # different FFTLog
#     p2 = ccl.halos.HaloProfileHOD(concentration=cm, lMmin_0=12.)
#     p2.update_precision_fftlog(**{"plaw_fourier": -2.0})
#     assert p1 != p2
=======
    # equivalent profiles
    cm2 = ccl.halos.ConcentrationDuffy08(mass_def='200m')
    p2 = ccl.halos.HaloProfileHOD(mass_def='200m',
                                  concentration=cm2, log10Mmin_0=12.)
    assert p1 == p2

    # different parameters
    p2 = ccl.halos.HaloProfileHOD(mass_def='200m', concentration=cm,
                                  log10Mmin_0=11.)
    assert p1 != p2

    # different mass-concentration
    cm2 = ccl.halos.ConcentrationConstant()
    p2 = ccl.halos.HaloProfileHOD(mass_def=cm2.mass_def, concentration=cm2,
                                  log10Mmin_0=12.)
    assert p1 != p2

    # different mass-concentration mass definition
    cm2 = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p2 = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=cm2,
                                  log10Mmin_0=12.)
    assert p1 != p2

    # different FFTLog
    p2 = ccl.halos.HaloProfileHOD(mass_def='200m', concentration=cm,
                                  log10Mmin_0=12.)
    p2.update_precision_fftlog(**{"plaw_fourier": -2.0})
    assert p1 != p2

    # different FFTLog type
    assert p1.precision_fftlog != 1
>>>>>>> master


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileNFW,
                          ccl.halos.HaloProfileHernquist,
                          ccl.halos.HaloProfileEinasto])
def test_empirical_smoke(prof_class):
<<<<<<< HEAD
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    with pytest.raises(TypeError):
        p = prof_class(concentration=None)
=======
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
>>>>>>> master

    if prof_class in [ccl.halos.HaloProfileNFW,
                      ccl.halos.HaloProfileHernquist]:
        with pytest.raises(ValueError):
<<<<<<< HEAD
            p = prof_class(concentration=c,
                           projected_analytic=True,
                           truncated=True)
        with pytest.raises(ValueError):
            p = prof_class(concentration=c,
                           cumul2d_analytic=True,
                           truncated=True)
        p = prof_class(concentration=c)
=======
            p = prof_class(mass_def=c.mass_def, concentration=c,
                           projected_analytic=True,
                           truncated=True)
        with pytest.raises(ValueError):
            p = prof_class(mass_def=c.mass_def, concentration=c,
                           cumul2d_analytic=True,
                           truncated=True)
        p = prof_class(mass_def=c.mass_def, concentration=c)
>>>>>>> master
        smoke_assert_prof_real(p, method='_fourier_analytic')
        smoke_assert_prof_real(p, method='_projected_analytic')
        smoke_assert_prof_real(p, method='_cumul2d_analytic')
    else:
        with pytest.raises(ValueError):
            p = prof_class(mass_def=c.mass_def, concentration=c,
                           projected_quad=True,
                           truncated=True)
        p = prof_class(mass_def=c.mass_def, concentration=c)
        smoke_assert_prof_real(p, method='_projected_quad')

<<<<<<< HEAD
    p = prof_class(concentration=c)
=======
    p = prof_class(mass_def=c.mass_def, concentration=c)
>>>>>>> master
    smoke_assert_prof_real(p, method='real')
    smoke_assert_prof_real(p, method='projected')
    smoke_assert_prof_real(p, method='fourier')


<<<<<<< HEAD
def test_empirical_smoke_CIB():
    with pytest.raises(TypeError):
        ccl.halos.HaloProfileCIBShang12(concentration=None, nu_GHz=417)


def test_cib_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=217)
=======
def test_cib_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=217,
                                        mass_def='200c')
>>>>>>> master
    beta_old = p.beta
    smoke_assert_prof_real(p, method='_real')
    smoke_assert_prof_real(p, method='_fourier')
    smoke_assert_prof_real(p, method='_fourier_variance')
    p.update_parameters(alpha=1.24, T0=20.0)
    assert p.alpha == 1.24
    assert p.T0 == 20.0
    assert p.beta == beta_old
    for n in ['alpha', 'T0', 'beta', 'gamma',
              's_z', 'Mmin', 'L0', 'siglog10M']:
        p.update_parameters(**{n: 1234.})
        assert getattr(p, n) == 1234.


<<<<<<< HEAD
def test_cib_raises():
    with pytest.raises(TypeError):
        ccl.halos.HaloProfileCIBShang12(concentration="my_conc", nu_GHz=217)


def test_cib_2pt_raises():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p_cib = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=217)
    p_tSZ = ccl.halos.HaloProfilePressureGNFW()
    p2pt = ccl.halos.Profile2ptCIB()
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_tSZ,
                         mass_def=M200)
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_cib,
                         prof2=p_tSZ, mass_def=M200)


def test_einasto_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileEinasto(concentration=c)
=======
def test_cib_2pt_diag():
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p1 = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=217,
                                         mass_def='200c')
    p2 = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=190,
                                         mass_def='200c')
    p2pt = ccl.halos.Profile2ptCIB()

    # Test diag=False
    F = p2pt.fourier_2pt(COSMO, 1., 1E13, 1., p1, prof2=p2, diag=False)
    assert np.ndim(F) == 0
    F = p2pt.fourier_2pt(COSMO, [1., 2], 1E13, 1., p1, prof2=p2, diag=False)
    assert F.shape == (2, 2)
    F = p2pt.fourier_2pt(COSMO, [1., 2], [1e12, 5e12, 1e13], 1., p1, prof2=p2,
                         diag=False)
    assert F.shape == (3, 2, 2)
    F2 = ccl.halos.Profile2pt().fourier_2pt(COSMO, [1., 2],
                                            [1e12, 5e12, 1e13], 1., p1,
                                            prof2=p2, diag=False)
    assert np.all(F == F2)


def test_cib_2pt_raises():
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p_cib = ccl.halos.HaloProfileCIBShang12(concentration=c, nu_GHz=217,
                                            mass_def='200c')
    p_tSZ = ccl.halos.HaloProfilePressureGNFW(mass_def='200c')
    p2pt = ccl.halos.Profile2ptCIB()
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_tSZ)
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_cib, prof2=p_tSZ)


def test_einasto_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p = ccl.halos.HaloProfileEinasto(mass_def='200c', concentration=c)
>>>>>>> master
    for M in [1E14, [1E14, 1E15]]:
        alpha_from_cosmo = p._get_alpha(COSMO, M, 1)

        p.update_parameters(alpha=1.)
        alpha = p._get_alpha(COSMO, M, 1)
        assert np.ndim(M) == np.ndim(alpha)
        assert np.all(p._get_alpha(COSMO, M, 1) == np.full_like(M, 1.))

        p.update_parameters(alpha='cosmo')
        assert np.ndim(M) == np.ndim(alpha_from_cosmo)
        assert np.all(p._get_alpha(COSMO, M, 1) == alpha_from_cosmo)


def test_gnfw_smoke():
    p = ccl.halos.HaloProfilePressureGNFW(mass_def='200c')
    beta_old = p.beta
    smoke_assert_prof_real(p)
    p.update_parameters(mass_bias=0.7,
                        alpha=1.24)
    assert p.alpha == 1.24
    assert p.mass_bias == 0.7
    assert p.beta == beta_old
    for n in ['P0', 'P0_hexp', 'alpha',
              'beta', 'gamma', 'alpha_P',
              'c500', 'mass_bias', 'x_out']:
        p.update_parameters(**{n: 1.314159})
        assert getattr(p, n) == 1.314159


def test_gnfw_refourier():
    p = ccl.halos.HaloProfilePressureGNFW(mass_def='200c')
    # Create Fourier template
    p._integ_interp()
<<<<<<< HEAD
    p_f1 = p.fourier(COSMO, 1., 1E13, 1., mass_def=M500c)
    # Check the Fourier profile gets recalculated
    p.update_parameters(alpha=1.32, c500=p.c500+0.1)
    p_f2 = p.fourier(COSMO, 1., 1E13, 1., mass_def=M500c)
=======
    p_f1 = p.fourier(COSMO, 1., 1E13, 1)
    # Check the Fourier profile gets recalculated
    p.update_parameters(alpha=1.32, c500=p.c500+0.1)
    p_f2 = p.fourier(COSMO, 1., 1E13, 1)
>>>>>>> master
    assert p_f1 != p_f2


def test_hod_smoke():
    prof_class = ccl.halos.HaloProfileHOD
<<<<<<< HEAD
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)

    with pytest.raises(TypeError):
        p = prof_class(concentration=None)

    p = prof_class(concentration=c)
=======
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')

    p = prof_class(mass_def=c.mass_def, concentration=c)
>>>>>>> master
    smoke_assert_prof_real(p)
    smoke_assert_prof_real(p, method='_usat_real')
    smoke_assert_prof_real(p, method='_usat_fourier')
    smoke_assert_prof_real(p, method='_fourier')
    smoke_assert_prof_real(p, method='_fourier_variance')
    for n in ['log10Mmin_0', 'log10Mmin_p', 'log10M0_0', 'log10M0_p',
              'log10M1_0', 'log10M1_p', 'siglnM_0', 'siglnM_p',
              'fc_0', 'fc_p', 'alpha_0', 'alpha_p',
              'bg_0', 'bg_p', 'bmax_0', 'bmax_p',
              'a_pivot']:
        p.update_parameters(**{n: 1234.})
        assert getattr(p, n) == 1234.


@pytest.mark.parametrize('real_prof', [True, False])
def test_hod_ns_independent(real_prof):
    def func(prof):
        return prof._real if real_prof else prof._fourier

<<<<<<< HEAD
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    hmd = c.mass_def
    p1 = ccl.halos.HaloProfileHOD(concentration=c,
                                  lMmin_0=12.,
                                  ns_independent=False)
    p2 = ccl.halos.HaloProfileHOD(concentration=c,
                                  lMmin_0=12.,
                                  ns_independent=True)
    # M < Mmin
    f1 = func(p1)(COSMO, 0.01, 1e10, 1., mass_def=hmd)
    assert np.all(f1 == 0)
    f2 = func(p2)(COSMO, 0.01, 1e10, 1., mass_def=hmd)
    assert np.all(f2 > 0)
    # M > Mmin
    f1 = func(p1)(COSMO, 0.01, 1e14, 1., mass_def=hmd)
    f2 = func(p2)(COSMO, 0.01, 1e14, 1., mass_def=hmd)
    assert np.allclose(f1, f2, rtol=0)
    # M == Mmin
    f1 = func(p1)(COSMO, 0.01, 1e12, 1., mass_def=hmd)
    f2 = func(p2)(COSMO, 0.01, 1e12, 1., mass_def=hmd)
    assert np.allclose(2*f1, f2+0.5, rtol=0)

    if not real_prof:
        f1 = p1._fourier_variance(COSMO, 0.01, 1e10, 1., mass_def=hmd)
        f2 = p2._fourier_variance(COSMO, 0.01, 1e10, 1., mass_def=hmd)
=======
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p1 = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=c,
                                  log10Mmin_0=12.,
                                  ns_independent=False)
    p2 = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=c,
                                  log10Mmin_0=12.,
                                  ns_independent=True)
    # M < Mmin
    f1 = func(p1)(COSMO, 0.01, 1e10, 1.)
    assert np.all(f1 == 0)
    f2 = func(p2)(COSMO, 0.01, 1e10, 1.)
    assert np.all(f2 > 0)
    # M > Mmin
    f1 = func(p1)(COSMO, 0.01, 1e14, 1.)
    f2 = func(p2)(COSMO, 0.01, 1e14, 1.)
    assert np.allclose(f1, f2, rtol=0)
    # M == Mmin
    f1 = func(p1)(COSMO, 0.01, 1e12, 1.)
    f2 = func(p2)(COSMO, 0.01, 1e12, 1.)
    assert np.allclose(2*f1, f2+0.5, rtol=0)

    if not real_prof:
        f1 = p1._fourier_variance(COSMO, 0.01, 1e10, 1)
        f2 = p2._fourier_variance(COSMO, 0.01, 1e10, 1)
>>>>>>> master
        assert f2 > f1 == 0

    p1.update_parameters(ns_independent=True)
    assert p1.ns_independent is True


@pytest.mark.parametrize("ns_indep", [True, False])
def test_hod_normalization(ns_indep):
    # Test that the HOD normalization works as expected.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    a_arr = np.linspace(0.5, 1.0, 8)
    hmc = ccl.halos.HMCalculator(
        mass_function="Tinker10", halo_bias="Tinker10", mass_def="200c")
    cm = ccl.halos.Concentration.create_instance(
        "Duffy08", mass_def=hmc.mass_def)
    prof = ccl.halos.HaloProfileHOD(mass_def='200c',
                                    concentration=cm, ns_independent=ns_indep)

    norm = np.array([prof.get_normalization(cosmo, a, hmc=hmc) for a in a_arr])

    def profile_norm(a):
        hmc._get_ingredients(cosmo, a, get_bf=False)
        uk0 = prof.fourier(cosmo, k=1e-5, M=hmc._mass, a=a).T
        return hmc._integrate_over_mf(uk0)

    norm_fourier = np.array([profile_norm(a) for a in a_arr])
    assert np.allclose(norm, norm_fourier, atol=0, rtol=1e-5)


def test_hod_2pt():
<<<<<<< HEAD
    pbad = ccl.halos.HaloProfilePressureGNFW()
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    pgood = ccl.halos.HaloProfileHOD(concentration=c)
    pgood_b = ccl.halos.HaloProfileHOD(concentration=c)
    p2 = ccl.halos.Profile2ptHOD()
    F0 = p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood,
                        prof2=pgood,
                        mass_def=M200)
    assert np.allclose(p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood,
                                      prof2=None, mass_def=M200),
                       F0, rtol=0)

    with pytest.raises(TypeError):
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pbad,
                       mass_def=M200)

    # TODO: bring back when proper __eq__s are implemented
    # doesn't raise because profiles are equivalent
    # p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood,
    #                prof2=pgood, mass_def=M200)

    # p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood,
    #                prof2=pgood_b, mass_def=M200)

    with pytest.raises(ValueError):
        pgood_b.update_parameters(lM0_0=10.)
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood,
                       prof2=pgood_b, mass_def=M200)
=======
    pbad = ccl.halos.HaloProfilePressureGNFW(mass_def='200c')
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    pgood = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=c)
    pgood_b = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=c)
    p2 = ccl.halos.Profile2ptHOD()
    F0 = p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood, prof2=pgood)
    assert np.allclose(p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood, prof2=None),
                       F0, rtol=0)

    with pytest.raises(TypeError):
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pbad)

    # doesn't raise because profiles are equivalent
    p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood, prof2=pgood)

    p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood, prof2=pgood_b)

    with pytest.raises(ValueError):
        pgood_b.update_parameters(log10M0_0=10.)
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood, prof2=pgood_b)

    with pytest.raises(ValueError):
        p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood, prof2=pbad)

    # Test diag = False
    F = p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood, prof2=pgood, diag=False)
    assert np.ndim(F) == 0
    F = p2.fourier_2pt(COSMO, [1., 2], 1E13, 1., pgood, prof2=pgood,
                       diag=False)
    assert F.shape == (2, 2)
    F = p2.fourier_2pt(COSMO, [1., 2], [1e12, 5e12, 1e13], 1., pgood,
                       prof2=pgood, diag=False)
    assert F.shape == (3, 2, 2)
    F2 = ccl.halos.Profile2pt().fourier_2pt(COSMO, [1., 2],
                                            [1e12, 5e12, 1e13], 1., pgood,
                                            diag=False)
    assert np.all(F == F2)
>>>>>>> master

    with pytest.raises(ValueError):
        p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood,
                       prof2=pbad, mass_def=M200)


def test_2pt_rcorr_smoke():
<<<<<<< HEAD
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileNFW(concentration=c)
    F0 = ccl.halos.Profile2pt().fourier_2pt(COSMO, 1., 1e13, 1., p,
                                            mass_def=M200)
    p2pt = ccl.halos.Profile2pt(r_corr=0)
    F1 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, mass_def=M200)
    assert F0 == F1
    F2 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, prof2=p, mass_def=M200)
=======
    c = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p = ccl.halos.HaloProfileNFW(mass_def='200c', concentration=c)
    F0 = ccl.halos.Profile2pt().fourier_2pt(COSMO, 1., 1e13, 1., p)
    p2pt = ccl.halos.Profile2pt(r_corr=0)
    F1 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p)
    assert F0 == F1
    F2 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, prof2=p)
>>>>>>> master
    assert F1 == F2

    p2pt.update_parameters(r_corr=-1.)
    assert p2pt.r_corr == -1.
<<<<<<< HEAD
    F3 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, mass_def=M200)
    assert F3 == 0


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileGaussian,
                          ccl.halos.HaloProfilePowerLaw])
def test_simple_smoke(prof_class):
    def r_s(cosmo, M, a, mass_def):
        return mass_def.get_radius(cosmo, M, a)

    if prof_class == ccl.halos.HaloProfileGaussian:
        p = prof_class(r_scale=r_s, rho0=one_f)
    else:
        p = prof_class(r_scale=r_s, tilt=one_f)
    smoke_assert_prof_real(p)


def test_gaussian_accuracy():
    def fk(k):
        return np.pi**1.5 * np.exp(-k**2 / 4)

    p = ccl.halos.HaloProfileGaussian(r_scale=one_f, rho0=one_f)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1., mass_def=M200)
    fk_arr_pred = fk(k_arr)
    res = np.fabs(fk_arr - fk_arr_pred)
    assert np.all(res < 5E-3)


@pytest.mark.parametrize('alpha', [-1.2, -2., -2.8])
def test_projected_plaw_accuracy(alpha):
    from scipy.special import gamma

    prefac = (np.pi**0.5 * gamma(-(alpha + 1) / 2) /
              gamma(-alpha / 2))

    def s_r_t(rt):
        return prefac * rt**(1 + alpha)

    def alpha_f(cosmo, a):
        return alpha

    p = ccl.halos.HaloProfilePowerLaw(r_scale=one_f, tilt=alpha_f)
    p.update_precision_fftlog(plaw_fourier=alpha)

    rt_arr = np.logspace(-3, 2, 1024)
    srt_arr = p.projected(COSMO, rt_arr, 1., 1., mass_def=M200)
    srt_arr_pred = s_r_t(rt_arr)
    res = np.fabs(srt_arr / srt_arr_pred - 1)
    assert np.all(res < 5E-3)


@pytest.mark.parametrize('alpha', [-1.2, -2., -2.8])
def test_plaw_accuracy(alpha):
    from scipy.special import gamma

    prefac = (2.**(3+alpha) * np.pi**1.5 *
              gamma((3 + alpha) / 2) /
              gamma(-alpha / 2))

    def fk(k):
        return prefac / k**(3 + alpha)

    def alpha_f(cosmo, a):
        return alpha

    p = ccl.halos.HaloProfilePowerLaw(r_scale=one_f, tilt=alpha_f)
    p.update_precision_fftlog(plaw_fourier=alpha)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1., mass_def=M200)
    fk_arr_pred = fk(k_arr)
    res = np.fabs(fk_arr / fk_arr_pred - 1)
    assert np.all(res < 5E-3)
=======
    F3 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p)
    assert F3 == 0

    # Test diag = False
    F = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, diag=False)
    assert isinstance(F, float)
    F = p2pt.fourier_2pt(COSMO, [1., 2.], 1e13, 1., p, diag=False)
    assert F.shape == (2, 2)
    F = p2pt.fourier_2pt(COSMO, [1., 2.], [1e12, 5e12, 1e13], 1., p,
                         diag=False)
    assert F.shape == (3, 2, 2)

    # Errors
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 1., 1e13, 1., None)
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, prof2=0)
>>>>>>> master


def get_nfw(real=False, fourier=False):
    # Return a subclass of the NFW profile with or without fourier analytic.
    NFW = type("NFW", (ccl.halos.HaloProfileNFW,), {})
    if real:
        NFW._real = ccl.halos.HaloProfileNFW._real
    if fourier:
        NFW._fourier = ccl.halos.HaloProfileNFW._fourier_analytic
    return NFW


@pytest.mark.parametrize("use_analytic", [True, False])
def test_nfw_accuracy(use_analytic):
    from scipy.special import sici

    tol = 1E-10 if use_analytic else 5E-3
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = get_nfw(real=True, fourier=use_analytic)(concentration=cM)
    M = 1E14
    a = 0.5
    c = cM(COSMO, M, a)
    r_Delta = M200.get_radius(COSMO, M, a) / a
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p = get_nfw(real=True, fourier=use_analytic)(mass_def='200c',
                                                 concentration=cM)
    M = 1E14
    a = 0.5
    c = cM(COSMO, M, a)
    r_Delta = cM.mass_def.get_radius(COSMO, M, a) / a
>>>>>>> master
    r_s = r_Delta / c

    def fk(k):
        x = k * r_s
        Si1, Ci1 = sici((1 + c) * x)
        Si2, Ci2 = sici(x)
        P1 = 1 / (np.log(1+c) - c / (1 + c))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c * x)/((1 + c) * x)
        return M * P1 * (P2 - P3)

    k_arr = np.logspace(-2, 2, 256) / r_Delta
<<<<<<< HEAD
    fk_arr = p.fourier(COSMO, k_arr, M, a, mass_def=M200)
=======
    fk_arr = p.fourier(COSMO, k_arr, M, a)
>>>>>>> master
    fk_arr_pred = fk(k_arr)
    # Normalize to  1 at low k
    res = np.fabs((fk_arr - fk_arr_pred) / M)
    assert np.all(res < tol)


def test_nfw_f2r():
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p1 = ccl.halos.HaloProfileNFW(concentration=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = get_nfw(fourier=True)(concentration=cM)
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p1 = ccl.halos.HaloProfileNFW(mass_def='200c', concentration=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = get_nfw(fourier=True)(mass_def='200c',
                               concentration=cM)
>>>>>>> master
    p2.update_precision_fftlog(padding_hi_fftlog=1E3)

    M = 1E14
    a = 0.5
    r_Delta = cM.mass_def.get_radius(COSMO, M, a) / a

    r_arr = np.logspace(-2, 1, ) * r_Delta
<<<<<<< HEAD
    pr_1 = p1.real(COSMO, r_arr, M, a, mass_def=M200)
    pr_2 = p2.real(COSMO, r_arr, M, a, mass_def=M200)
=======
    pr_1 = p1.real(COSMO, r_arr, M, a)
    pr_2 = p2.real(COSMO, r_arr, M, a)
>>>>>>> master

    id_good = r_arr < r_Delta  # Profile is 0 otherwise
    res = np.fabs(pr_2[id_good] / pr_1[id_good] - 1)
    assert np.all(res < 0.01)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_projected_accuracy(fourier_analytic):
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileNFW(concentration=cM, truncated=False,
                                  projected_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(concentration=cM, truncated=False)
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileNFW(mass_def='200c', concentration=cM,
                                  truncated=False, projected_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(mass_def='200c',
                                           concentration=cM, truncated=False)
>>>>>>> master

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
<<<<<<< HEAD
    srt1 = p1.projected(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.projected(COSMO, rt, M, a, mass_def=M200)
=======
    srt1 = p1.projected(COSMO, rt, M, a)
    srt2 = p2.projected(COSMO, rt, M, a)
>>>>>>> master

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_cumul2d_accuracy(fourier_analytic):
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileNFW(concentration=cM, truncated=False,
                                  cumul2d_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(concentration=cM, truncated=False)
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileNFW(mass_def='200c', concentration=cM,
                                  truncated=False, cumul2d_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(mass_def='200c',
                                           concentration=cM, truncated=False)
>>>>>>> master

    M = 1E14
    a = 1.0
    rt = np.logspace(-3, 2, 1024)
<<<<<<< HEAD
    srt1 = p1.cumul2d(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.cumul2d(COSMO, rt, M, a, mass_def=M200)
=======
    srt1 = p1.cumul2d(COSMO, rt, M, a)
    srt2 = p2.cumul2d(COSMO, rt, M, a)
>>>>>>> master

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


def test_upd_fftlog_raises():
    # Verify that FFTLogParams is immutable unless changed in a control manner.
<<<<<<< HEAD
    prof = ccl.halos.HaloProfilePressureGNFW()
=======
    prof = ccl.halos.HaloProfilePressureGNFW(mass_def='200c')
>>>>>>> master
    new_params = {"hello_there": 0.}
    with pytest.raises(AttributeError):
        prof.update_precision_fftlog(**new_params)

    with pytest.raises(AttributeError):
        prof.precision_fftlog.plaw_projected = 1


@pytest.mark.parametrize("use_analytic", [True, False])
def test_hernquist_accuracy(use_analytic):
    from scipy.special import sici

    tol = 1E-10 if use_analytic else 5E-3
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileHernquist(concentration=cM,
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p = ccl.halos.HaloProfileHernquist(mass_def='200c', concentration=cM,
>>>>>>> master
                                       fourier_analytic=use_analytic)
    M = 1E14
    a = 0.5
    c = cM(COSMO, M, a)
<<<<<<< HEAD
    r_Delta = M200.get_radius(COSMO, M, a) / a
=======
    r_Delta = cM.mass_def.get_radius(COSMO, M, a) / a
>>>>>>> master
    r_s = r_Delta / c

    def fk(k):
        x = k * r_s
        cp1 = c + 1
        Si1, Ci1 = sici(cp1 * x)
        Si2, Ci2 = sici(x)
        P1 = 1 / ((c / cp1)**2 / 2)
        P2 = x * np.sin(x) * (Ci1 - Ci2) - x * np.cos(x) * (Si1 - Si2)
        P3 = (-1 + np.sin(c * x) / (cp1**2 * x)
              + cp1 * np.cos(c * x) / (cp1**2))
        return M * P1 * (P2 - P3) / 2

    k_arr = np.logspace(-2, 2, 256) / r_Delta
<<<<<<< HEAD
    fk_arr = p.fourier(COSMO, k_arr, M, a, mass_def=M200)
=======
    fk_arr = p.fourier(COSMO, k_arr, M, a)
>>>>>>> master
    fk_arr_pred = fk(k_arr)
    # Normalize to  1 at low k
    res = np.fabs((fk_arr - fk_arr_pred) / M)
    assert np.all(res < tol)


def test_hernquist_f2r():
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p1 = ccl.halos.HaloProfileHernquist(concentration=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = ccl.halos.HaloProfileHernquist(concentration=cM,
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    p1 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM,
>>>>>>> master
                                        fourier_analytic=True)
    with UnlockInstance(p2):
        # For the fourier computation, we require that the profile does not
        # have an implemented `_real` method. This is internally checked
        # by verifying that the hook `__islinkedabstractmethod__` is not there.
        # Here, we replace the bound method with a function that has this hook.
        p2._real = ccl.halos.HaloProfile._real
    p2.update_precision_fftlog(padding_hi_fftlog=1E3)

    M = 1E14
    a = 0.5
    r_Delta = cM.mass_def.get_radius(COSMO, M, a) / a

    r_arr = np.logspace(-2, 1, ) * r_Delta
<<<<<<< HEAD
    pr_1 = p1.real(COSMO, r_arr, M, a, mass_def=M200)
    pr_2 = p2.real(COSMO, r_arr, M, a, mass_def=M200)
=======
    pr_1 = p1.real(COSMO, r_arr, M, a)
    pr_2 = p2.real(COSMO, r_arr, M, a)
>>>>>>> master

    id_good = r_arr < r_Delta  # Profile is 0 otherwise
    res = np.fabs(pr_2[id_good] / pr_1[id_good] - 1)
    assert np.all(res < 0.01)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_hernquist_projected_accuracy(fourier_analytic):
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileHernquist(concentration=cM, truncated=False,
                                        projected_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(concentration=cM, truncated=False,
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM, truncated=False,
                                        projected_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM, truncated=False,
>>>>>>> master
                                        fourier_analytic=fourier_analytic)

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
<<<<<<< HEAD
    srt1 = p1.projected(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.projected(COSMO, rt, M, a, mass_def=M200)
=======
    srt1 = p1.projected(COSMO, rt, M, a)
    srt2 = p2.projected(COSMO, rt, M, a)
>>>>>>> master

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_hernquist_cumul2d_accuracy(fourier_analytic):
<<<<<<< HEAD
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileHernquist(concentration=cM, truncated=False,
                                        cumul2d_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(concentration=cM, truncated=False,
=======
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM, truncated=False,
                                        cumul2d_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(mass_def='200c',
                                        concentration=cM, truncated=False,
>>>>>>> master
                                        fourier_analytic=fourier_analytic)
    M = 1E14
    a = 1.0
    rt = np.logspace(-3, 2, 1024)
<<<<<<< HEAD
    srt1 = p1.cumul2d(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.cumul2d(COSMO, rt, M, a, mass_def=M200)
=======
    srt1 = p1.cumul2d(COSMO, rt, M, a)
    srt2 = p2.cumul2d(COSMO, rt, M, a)
>>>>>>> master

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


<<<<<<< HEAD
=======
def test_einasto_projected_accuracy():
    cM = ccl.halos.ConcentrationDuffy08(mass_def='200c')
    # projected profile from quad
    p1 = ccl.halos.HaloProfileEinasto(mass_def='200c',
                                      concentration=cM, truncated=False,
                                      projected_quad=True)
    # projected profile from FFTLog
    p2 = ccl.halos.HaloProfileEinasto(mass_def='200c',
                                      concentration=cM, truncated=False,
                                      projected_quad=False)
    # truncated projected profile from FFTLog
    p3 = ccl.halos.HaloProfileEinasto(mass_def='200c',
                                      concentration=cM, truncated=True,
                                      projected_quad=False)

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.projected(COSMO, rt, M, a)[:500]
    srt2 = p2.projected(COSMO, rt, M, a)[:500]
    srt3 = p3.projected(COSMO, rt, M, a)[:500]

    res1 = np.fabs(srt2/srt1-1)
    res2 = np.fabs(srt3/srt1-1)
    assert np.all(res1 < 6e-5)
    assert np.all(res2 < 6e-2)


>>>>>>> master
def test_HaloProfile_abstractmethods():
    # Test that `HaloProfile` and its subclasses can't be instantiated if
    # either `_real` or `_fourier` have not been defined.
    with pytest.raises(TypeError):
        ccl.halos.HaloProfile()
<<<<<<< HEAD
=======


def test_IA_update():
    cM = ccl.halos.ConcentrationDuffy08(mass_def="200m")
    p = ccl.halos.SatelliteShearHOD(concentration=cM, mass_def="200m")
    for attr in ['a1h', 'b', 'log10Mmin_0', 'log10Mmin_p',
                 'log10M0_0', 'log10M0_p', 'log10M1_0', 'log10M1_p',
                 'siglnM_0', 'siglnM_p', 'alpha_0', 'alpha_p',
                 'bg_0', 'bg_p', 'bmax_0', 'bmax_p', 'a_pivot',
                 'rmin', 'N_r', 'N_jn']:
        p.update_parameters(**{attr: -123})
        assert getattr(p, attr) == -123
    # Special cases
    p.update_parameters(lmax=5)
    assert p.lmax == 5
    p.update_parameters(ns_independent=True)
    assert p.ns_independent


def test_IA_profile():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                          sigma8=0.83, n_s=0.96)
    k_arr = np.geomspace(1E-3, 1e3, 256)  # For evaluating
    cM = ccl.halos.ConcentrationDuffy08(mass_def="200m")

    # lmax too low
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.SatelliteShearHOD(concentration=cM, lmax=1,
                                    mass_def="200m")

    # lmax too high
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.SatelliteShearHOD(concentration=cM, lmax=14,
                                    mass_def="200m")

    # lmax odd
    assert (ccl.halos.SatelliteShearHOD(concentration=cM, mass_def="200m",
                                        lmax=7).lmax) % 2 == 0

    # Run with b!={0,2}
    assert (ccl.halos.SatelliteShearHOD(
        concentration=cM, b=-1.9, mass_def="200m",
        lmax=12)._angular_fl).shape == (6, 1)

    # Testing FFTLog accuracy vs simps and spline method.
    s_g_HOD1 = ccl.halos.SatelliteShearHOD(concentration=cM,
                                           mass_def="200m")
    s_g_HOD2 = ccl.halos.SatelliteShearHOD(concentration=cM,
                                           mass_def="200m",
                                           integration_method='simpson')
    s_g_HOD3 = ccl.halos.SatelliteShearHOD(concentration=cM,
                                           mass_def="200m",
                                           integration_method='spline')
    s_g1 = s_g_HOD1._usat_fourier(cosmo, k_arr, 1e13, 1.)
    s_g2 = s_g_HOD2._usat_fourier(cosmo, k_arr, 1e13, 1.)
    s_g3 = s_g_HOD3._usat_fourier(cosmo, k_arr, 1e13, 1.)
    assert np.all(np.abs((s_g1 - s_g2) / s_g2)) > 0.05
    assert np.all(np.abs((s_g3 - s_g2) / s_g3)) > 0.05

    # Wrong integration method
    with pytest.raises(ValueError):
        ccl.halos.SatelliteShearHOD(concentration=cM,
                                    mass_def="200m",
                                    integration_method="something_else")


def test_prefactor():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                          sigma8=0.83, n_s=0.96)
    cM = ccl.halos.ConcentrationDuffy08(mass_def="200m")
    nM = ccl.halos.MassFuncTinker08(mass_def="200m")
    bM = ccl.halos.HaloBiasTinker10(mass_def="200m")
    hmc = ccl.halos.HMCalculator(mass_function=nM,
                                 halo_bias=bM, mass_def="200m")

    p = ccl.halos.HaloProfilePressureGNFW(mass_def="200m")  # a simple profile
    assert np.all(np.abs(p.get_normalization(cosmo, 1., hmc=hmc)-1.0 < 1e-10))

    p = ccl.halos.SatelliteShearHOD(concentration=cM, mass_def="200m")
    assert p.get_normalization(cosmo, 1., hmc=hmc) > 0
>>>>>>> master
