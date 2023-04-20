import numpy as np
import pyccl as ccl
import pytest
from pyccl import UnlockInstance
from pyccl.nl_pt import PTCalculator, get_pt_pk2d


# TODO v3: deprecate these tests

def catch_warning(func):
    def wrapper(*args, **kwargs):
        with pytest.warns(ccl.CCLDeprecationWarning):
            return func(*args, **kwargs)
    return wrapper


PTCalculator.__init__ = catch_warning(PTCalculator.__init__)
get_pt_pk2d = catch_warning(get_pt_pk2d)


NZ = 128
ZZ = np.linspace(0., 1., NZ)
BZ_C = 2.
BZ = BZ_C * np.ones(NZ)
COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
TRS = {'TG': ccl.nl_pt.PTNumberCountsTracer((ZZ, BZ),
                                            (ZZ, BZ),
                                            (ZZ, BZ)),
       'TI': ccl.nl_pt.PTIntrinsicAlignmentTracer((ZZ, BZ),
                                                  (ZZ, BZ),
                                                  (ZZ, BZ)),
       'TM': ccl.nl_pt.PTMatterTracer()}
PTC = PTCalculator(with_NC=True, with_IA=True, with_dd=True)

a_1 = a_2 = a_d = 1.0
gz = ccl.growth_factor(COSMO, 1./(1+ZZ))
ZZ_1 = 1.0
gz_1 = ccl.growth_factor(COSMO, 1./(1+ZZ_1))
a_1_v = a_1*np.ones_like(ZZ)
Om_m = COSMO['Omega_m']
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_m = ccl.physical_constants.RHO_CRITICAL * COSMO['Omega_m']
Om_m_fid = 0.3

c_1_t = -1*a_1*5e-14*rho_crit*COSMO['Omega_m']/gz
c_1_t_1 = -1*a_1*5e-14*rho_crit*COSMO['Omega_m']/gz_1
c_d_t = -1*a_d*5e-14*rho_crit*COSMO['Omega_m']/gz
c_2_t = a_2*5*5e-14*rho_crit*COSMO['Omega_m']**2/(Om_m_fid*gz**2)
c_2_t_des = a_2*5*5e-14*rho_crit*COSMO['Omega_m']/(gz**2)

ks = np.logspace(-3, 2, 512)


def test_pt_workspace_smoke():
    w = PTCalculator(log10k_min=-3, log10k_max=1,
                     nk_per_decade=10, pad_factor=2)
    assert len(w.ks) == 40


@pytest.mark.parametrize('options', [['TG', 'TG', False, False, PTC],
                                     ['TG', 'TG', False, False, None],
                                     ['TG', 'TG', False, True, PTC],
                                     ['TG', 'TI', False, False, PTC],
                                     ['TG', 'TM', False, False, PTC],
                                     ['TI', 'TG', False, False, PTC],
                                     ['TI', 'TI', False, False, PTC],
                                     ['TI', 'TI', True, False, PTC],
                                     ['TI', 'TM', False, False, PTC],
                                     ['TM', 'TG', False, False, PTC],
                                     ['TM', 'TI', False, False, PTC],
                                     ['TM', 'TM', False, False, PTC]])
def test_pt_get_pk2d_smoke(options):
    if options[0] == options[1]:
        t2 = None
    else:
        t2 = TRS[options[1]]
    pk = get_pt_pk2d(COSMO,
                     TRS[options[0]],
                     tracer2=t2,
                     return_ia_bb=options[2],
                     sub_lowk=options[3],
                     ptc=options[4])
    assert isinstance(pk, ccl.Pk2D)


def test_pt_pk2d_bb():
    pee = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC)
    pbb = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC, return_ia_bb=True)
    pee2, pbb2 = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                             return_ia_ee_and_bb=True)
    assert pee(0.1, 0.9, COSMO) == pee2(0.1, 0.9, COSMO)
    assert pbb(0.1, 0.9, COSMO) == pbb2(0.1, 0.9, COSMO)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'spt'])
def test_pt_get_pk2d_nl(nl):
    pk = get_pt_pk2d(COSMO, TRS['TG'], nonlin_pk_type=nl)
    assert isinstance(pk, ccl.Pk2D)


def test_ptc_raises():
    with pytest.raises(ValueError):
        PTC.update_pk(np.zeros(4))


@pytest.mark.parametrize('typ_nlin,typ_nloc', [('linear', 'nonlinear'),
                                               ('nonlinear', 'linear'),
                                               ('nonlinear', 'spt')])
def test_k2pk_types(typ_nlin, typ_nloc):
    tg = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0., bk2=1.)
    tm = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0.)
    pkmm = get_pt_pk2d(COSMO, tm, tracer2=tm, ptc=PTC,
                       nonlin_pk_type=typ_nlin)
    pkmm_t = get_pt_pk2d(COSMO, tm, tracer2=tm, ptc=PTC,
                         nonlin_pk_type=typ_nloc)
    pkgg = get_pt_pk2d(COSMO, tg, tracer2=tg, ptc=PTC,
                       nonloc_pk_type=typ_nloc)
    ks = np.geomspace(1E-3, 1E1, 128)
    p1 = pkgg(ks, 1., COSMO)
    p2 = pkmm(ks, 1., COSMO)+ks**2*pkmm_t(ks, 1., COSMO)
    assert np.all(np.fabs(p1/p2-1) < 1E-4)


def test_k2pk():
    # Tests the k2 term scaling
    ptc = PTCalculator(with_NC=True)

    zs = np.array([0., 1.])
    gs4 = ccl.growth_factor(COSMO, 1./(1+zs))**4
    pk_lin_z0 = ccl.linear_matter_power(COSMO, ptc.ks, 1.)
    ptc.update_pk(pk_lin_z0)

    Pd1d1 = np.array([ccl.linear_matter_power(COSMO, ptc.ks, a)
                      for a in 1./(1+zs)]).T
    one = np.ones_like(zs)
    zero = np.zeros_like(zs)
    pmm = ptc.get_pgg(Pd1d1, gs4, one, zero, zero,
                      one, zero, zero, True)
    pmm_b = ptc.get_pgm(Pd1d1, gs4, one, zero, zero)
    pmk = ptc.get_pgg(Pd1d1, gs4, one, zero, zero,
                      one, zero, zero, True, bk21=one)
    pmk_b = ptc.get_pgm(Pd1d1, gs4, one, zero, zero, bk2=one)
    pkk = ptc.get_pgg(Pd1d1, gs4, one, zero, zero, one, zero, zero, True,
                      bk21=one, bk22=one)
    ks = ptc.ks[:, None]
    assert np.all(np.fabs(pmm/Pd1d1-1) < 1E-10)
    assert np.all(np.fabs(pmm_b/Pd1d1-1) < 1E-10)
    assert np.all(np.fabs(pmk/(pmm*(1+0.5*ks**2))-1) < 1E-10)
    assert np.all(np.fabs(pmk_b/(pmm*(1+0.5*ks**2))-1) < 1E-10)
    assert np.all(np.fabs(pkk/(pmm*(1+ks**2))-1) < 1E-10)


def test_pk_cutoff():
    # Tests the exponential cutoff
    ptc1 = PTCalculator(with_NC=True)
    ptc2 = PTCalculator(with_NC=True, k_cutoff=10., n_exp_cutoff=2.)
    zs = np.array([0., 1.])
    gs4 = ccl.growth_factor(COSMO, 1./(1+zs))**4
    pk_lin_z0 = ccl.linear_matter_power(COSMO, ptc1.ks, 1.)
    ptc1.update_pk(pk_lin_z0)
    ptc2.update_pk(pk_lin_z0)

    Pd1d1 = np.array([ccl.linear_matter_power(COSMO, ptc1.ks, a)
                      for a in 1./(1+zs)]).T
    one = np.ones_like(zs)
    zero = np.zeros_like(zs)
    p1 = ptc1.get_pgg(Pd1d1, gs4, one, zero, zero,
                      one, zero, zero, True).T
    p2 = ptc2.get_pgg(Pd1d1, gs4, one, zero, zero,
                      one, zero, zero, True).T
    expcut = np.exp(-(ptc1.ks/10.)**2)
    assert np.all(np.fabs(p1*expcut/p2-1) < 1E-10)


def test_pt_get_pk2d_raises():
    # Wrong tracer type 2
    with pytest.raises(TypeError):
        get_pt_pk2d(COSMO, TRS['TG'], tracer2=3, ptc=PTC)
    # Wrong tracer type 1
    with pytest.raises(TypeError):
        get_pt_pk2d(COSMO, 3, tracer2=TRS['TG'], ptc=PTC)
    # Wrong calculator type
    with pytest.raises(TypeError):
        get_pt_pk2d(COSMO, TRS['TG'], ptc=3)

    # Incomplete calculator
    ptc_empty = PTCalculator(with_NC=False, with_IA=False, with_dd=False)
    for t in ['TG', 'TI', 'TM']:
        with pytest.raises(ValueError):
            get_pt_pk2d(COSMO, TRS[t], nonlin_pk_type='spt', ptc=ptc_empty)

    # Wrong non-linear Pk
    with pytest.raises(NotImplementedError):
        get_pt_pk2d(COSMO, TRS['TM'], nonlin_pk_type='halofat')

    # Wrong tracer types
    tdum = ccl.nl_pt.PTMatterTracer()
    with UnlockInstance(tdum):
        tdum.type = 'A'
    for t in ['TG', 'TI', 'TM']:
        with pytest.raises(NotImplementedError):
            get_pt_pk2d(COSMO, TRS[t], tracer2=tdum, ptc=PTC)
    with pytest.raises(NotImplementedError):
        get_pt_pk2d(COSMO, tdum, tracer2=TRS['TM'], ptc=PTC)


def test_translate_IA_norm():
    # test that it works with scalar a, vector z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, z=ZZ, a1=a_1,
                                                a1delta=a_d, a2=a_2,
                                                Om_m2_for_c2=False)
    assert c_1.all() == c_1_t.all()
    assert c_d.all() == c_d_t.all()
    assert c_2.all() == c_2_t_des.all()

    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, z=ZZ, a1=a_1,
                                                a1delta=a_d, a2=a_2,
                                                Om_m2_for_c2=True)
    assert c_2.all() == c_2_t.all()

    # test that it works with scalar a, scalar z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, z=ZZ_1, a1=a_1,
                                                Om_m2_for_c2=False)
    assert c_1 == c_1_t_1

    # test that it works with vector a, vector z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, z=ZZ, a1=a_1_v,
                                                Om_m2_for_c2=False)
    assert c_1.all() == c_1_t.all()


def test_return_ptc():
    # if no ptc is input, check that returned pk and ptc objects have
    # the correct properties.
    pk, ptc1 = get_pt_pk2d(COSMO, TRS['TG'], return_ptc=True)
    assert isinstance(pk, ccl.Pk2D)
    assert isinstance(ptc1, PTCalculator)
    # same test with EE/BB output
    pee2, pbb2, ptc2 = get_pt_pk2d(COSMO, TRS['TI'],
                                   return_ia_ee_and_bb=True,
                                   return_ptc=True)
    assert isinstance(pee2, ccl.Pk2D)
    assert isinstance(pbb2, ccl.Pk2D)
    assert isinstance(ptc2, PTCalculator)
    # check that returned ptc matches input ptc.
    pk_2, ptc1_2 = get_pt_pk2d(COSMO, TRS['TG'], ptc=PTC, return_ptc=True)
    pee2_2, pbb2_2, ptc2_2 = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                                         return_ia_ee_and_bb=True,
                                         return_ptc=True)
    assert ptc1_2 is PTC
    assert ptc2_2 is PTC
    # check that the result outputs are the same
    # for the internally initialized ptc.
    assert np.allclose(pk_2(ks, 1., COSMO), pk(ks, 1., COSMO))
    assert np.allclose(pee2_2(ks, 1., COSMO), pee2(ks, 1., COSMO))
    assert np.allclose(pbb2_2(ks, 1., COSMO), pbb2(ks, 1., COSMO))


def test_pt_no_ptc_update():
    pee1 = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC)
    pee1_no_update = get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC, update_ptc=False)

    COSMO2 = ccl.Cosmology(
        Omega_c=COSMO["Omega_c"], Omega_b=COSMO["Omega_b"], h=COSMO["h"],
        sigma8=COSMO["sigma8"]+0.05, n_s=COSMO["n_s"],
        transfer_function='bbks', matter_power_spectrum='linear')

    pee2_no_update = get_pt_pk2d(COSMO2, TRS['TI'], ptc=PTC, update_ptc=False)
    pee2 = get_pt_pk2d(COSMO2, TRS['TI'], ptc=PTC)

    assert pee1(0.1, 0.9, COSMO) == pee1_no_update(0.1, 0.9, COSMO)
    assert pee2(0.1, 0.9, COSMO) != pee2_no_update(0.1, 0.9, COSMO2)
