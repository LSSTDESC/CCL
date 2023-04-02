import numpy as np
import pyccl as ccl
import pytest


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
PTC = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                     with_matter_1loop=True)
PTC.update_ingredients(COSMO)

#ks = np.logspace(-3, 2, 512)


def test_pt_tracer_smoke():
    ccl.nl_pt.PTTracer()


def test_pt_tracer_m_smoke():
    ccl.nl_pt.PTMatterTracer()


@pytest.mark.parametrize('b2', [(ZZ, BZ), BZ_C, None])
def test_pt_tracer_nc_smoke(b2):
    pt_tr = ccl.nl_pt.PTNumberCountsTracer((ZZ, BZ),
                                           b2=b2,
                                           bs=(ZZ, BZ))

    # Test b1 and bs do the right thing
    for b in [pt_tr.b1, pt_tr.bs]:
        assert b(0.2) == BZ_C

    # Test b2 does the right thing
    if b2 is not None:
        assert pt_tr.b2(0.2) == BZ_C
        zz = np.array([0.2])
        assert pt_tr.b2(zz).squeeze() == BZ_C


@pytest.mark.parametrize('c2', [(ZZ, BZ), BZ_C, None])
def test_pt_tracer_ia_smoke(c2):
    pt_tr = ccl.nl_pt.PTIntrinsicAlignmentTracer((ZZ, BZ),
                                                 c2=c2,
                                                 cdelta=(ZZ, BZ))

    # Test c1 and cdelta do the right thing
    for b in [pt_tr.c1, pt_tr.cdelta]:
        assert b(0.2) == BZ_C

    # Test c2 does the right thing
    if c2 is not None:
        assert pt_tr.c2(0.2) == BZ_C
        zz = np.array([0.2])
        assert pt_tr.c2(zz).squeeze() == BZ_C


def test_pt_tracer_get_bias():
    pt_tr = ccl.nl_pt.PTNumberCountsTracer((ZZ, BZ),
                                           b2=(ZZ, BZ),
                                           bs=(ZZ, BZ))
    b = pt_tr.get_bias('b1', 0.1)
    assert b == BZ_C

    with pytest.raises(KeyError):
        pt_tr.get_bias('b_one', 0.1)


def test_pt_calculator_smoke():
    c = ccl.nl_pt.EulerianPTCalculator(log10k_min=-3,
                                       log10k_max=1,
                                       nk_per_decade=10,
                                       extra_params={'pad_factor': 2})
    assert len(c.k_s) == 40


@pytest.mark.parametrize('options', [['TG', 'TG', False, False],
                                     ['TG', 'TG', False, True],
                                     ['TG', 'TI', False, False],
                                     ['TG', 'TM', False, False],
                                     ['TI', 'TG', False, False],
                                     ['TI', 'TI', False, False],
                                     ['TI', 'TI', True, False],
                                     ['TI', 'TM', False, False],
                                     ['TM', 'TG', False, False],
                                     ['TM', 'TI', False, False],
                                     ['TM', 'TM', False, False]])
def test_pt_get_pk2d_smoke(options):
    if options[0] == options[1]:
        t2 = None
    else:
        t2 = TRS[options[1]]
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        extra_params={'sub_lowk': options[3]})
    ptc.update_ingredients(COSMO)
    pk = ptc.get_pk2d_biased(TRS[options[0]],
                             tracer2=t2,
                             return_ia_bb=options[2])
    assert isinstance(pk, ccl.Pk2D)


def test_pt_pk2d_bb_smoke():
    pee = PTC.get_pk2d_biased(TRS['TI'])
    pbb = PTC.get_pk2d_biased(TRS['TI'], return_ia_bb=True)
    assert pee.eval(0.1, 0.9, COSMO) != pbb.eval(0.1, 0.9, COSMO)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'pt'])
def test_pt_get_pk2d_nl(nl):
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=nl, bk_pk_kind=nl)
    ptc.update_ingredients(COSMO)
    pk = ptc.get_pk2d_biased(TRS['TG'])
    assert isinstance(pk, ccl.Pk2D)


@pytest.mark.parametrize('typ_nlin,typ_nloc', [('linear', 'nonlinear'),
                                               ('nonlinear', 'linear'),
                                               ('nonlinear', 'pt')])
def test_k2pk_types(typ_nlin, typ_nloc):
    tg = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0., bk2=1.)
    tm = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0.)
    ptc1 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=typ_nlin, bk_pk_kind=typ_nloc)
    ptc1.update_ingredients(COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=typ_nloc)
    ptc2.update_ingredients(COSMO)
    pkmm = ptc1.get_pk2d_biased(tm, tracer2=tm)
    pkmm2 = ptc2.get_pk2d_biased(tm, tracer2=tm)
    pkgg = ptc1.get_pk2d_biased(tg, tracer2=tg)
    ks = np.geomspace(1E-3, 1E1, 128)
    p1 = pkgg.eval(ks, 1., COSMO)
    p2 = pkmm.eval(ks, 1., COSMO)+ks**2*pkmm2.eval(ks, 1., COSMO)
    assert np.all(np.fabs(p1/p2-1) < 1E-4)


@pytest.mark.parametrize('kind',
                         ['m:m', 'm:b1', 'm:b2', 'm:b3nl', 'm:bs',
                          'm:bk2', 'm:c1', 'm:c2', 'm:cdelta',
                          'b1:b1', 'b1:b2', 'b1:b3nl', 'b1:bs',
                          'b1:bk2', 'b1:c1', 'b1:c2', 'b1:cdelta',
                          'b2:b2', 'b2:b3nl', 'b2:bs',
                          'b2:bk2', 'b2:c1', 'b2:c2', 'b2:cdelta',
                          'b3nl:b3nl', 'b3nl:bs', 'b3nl:bk2',
                          'b3nl:c1', 'b3nl:c2', 'b3nl:cdelta',
                          'bs:bs', 'bs:bk2', 'bs:c1', 'bs:c2',
                          'bs:cdelta', 'bk2:bk2', 'bk2:c1', 'bk2:c2',
                          'bk2:cdelta', 'c1:c1', 'c1:c2', 'c1:cdelta',
                          'c2:c2', 'c2:cdelta', 'cdelta:cdelta'])
def test_deconstruction(kind):
    b_nc = ['b1', 'b2', 'b3nl', 'bs', 'bk2']
    b_ia = ['c1', 'c2', 'cdelta']
    pk1 = PTC.get_pk2d_template(kind)

    def get_tr(tn):
        if tn == 'm':
            return ccl.nl_pt.PTMatterTracer()
        if tn in b_nc:
            bdict = {b: 0.0 for b in b_nc}
            bdict[tn] = 1.0
            return ccl.nl_pt.PTNumberCountsTracer(
                b1=bdict['b1'], b2=bdict['b2'],
                bs=bdict['bs'], bk2=bdict['bk2'],
                b3nl=bdict['b3nl'])
        if tn in b_ia:
            bdict = {b: 0.0 for b in b_ia}
            bdict[tn] = 1.0
            return ccl.nl_pt.PTIntrinsicAlignmentTracer(
                c1=bdict['c1'], c2=bdict['c2'],
                cdelta=bdict['cdelta'])

    tn1, tn2 = kind.split(':')
    t1 = get_tr(tn1)
    t2 = get_tr(tn2)

    pk2 = PTC.get_pk2d_biased(t1, tracer2=t2)

    if pk1 is None:
        assert pk2.eval(0.5, 1.0, COSMO) == 0.0
    else:
        assert pk1.eval(0.5, 1.0, COSMO) == pk2.eval(0.5, 1.0, COSMO)
        # Check cached
        pk3 = PTC._pk2d_temp[PTC._pk_alias[kind]]
        assert pk1.eval(0.5, 1.0, COSMO) == pk3.eval(0.5, 1.0, COSMO)


@pytest.mark.parametrize('kind',
                         ['c2:c2', 'c2:cdelta', 'cdelta:cdelta'])
def test_deconstruction_bb(kind):
    b_ia = ['c1', 'c2', 'cdelta']
    pk1 = PTC.get_pk2d_template(kind, return_ia_bb=True)

    def get_tr(tn):
        bdict = {b: 0.0 for b in b_ia}
        bdict[tn] = 1.0
        return ccl.nl_pt.PTIntrinsicAlignmentTracer(
            c1=bdict['c1'], c2=bdict['c2'],
            cdelta=bdict['cdelta'])

    tn1, tn2 = kind.split(':')
    t1 = get_tr(tn1)
    t2 = get_tr(tn2)

    pk2 = PTC.get_pk2d_biased(t1, tracer2=t2, return_ia_bb=True)

    assert pk1.eval(0.5, 1.0, COSMO) == pk2.eval(0.5, 1.0, COSMO)


def atest_k2pk():
    # Tests the k2 term scaling
    ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True)
    ptc.update_ingredients(COSMO)

    zs = np.array([0., 1.])
    gs4 = ccl.growth_factor(COSMO, 1./(1+zs))**4
    pk_lin_z0 = ccl.linear_matter_power(COSMO, ptc.ks, 1.)

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


def atest_pk_cutoff():
    # Tests the exponential cutoff
    ptc1 = ccl.nl_pt.PTCalculator(with_NC=True)
    ptc2 = ccl.nl_pt.PTCalculator(with_NC=True,
                                  k_cutoff=10.,
                                  n_exp_cutoff=2.)
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


def atest_pt_get_pk2d_raises():
    # Wrong tracer type 2
    with pytest.raises(TypeError):
        ccl.nl_pt.get_pt_pk2d(COSMO,
                              TRS['TG'],
                              tracer2=3,
                              ptc=PTC)
    # Wrong tracer type 1
    with pytest.raises(TypeError):
        ccl.nl_pt.get_pt_pk2d(COSMO,
                              3,
                              tracer2=TRS['TG'],
                              ptc=PTC)
    # Wrong calculator type
    with pytest.raises(TypeError):
        ccl.nl_pt.get_pt_pk2d(COSMO,
                              TRS['TG'],
                              ptc=3)

    # Incomplete calculator
    ptc_empty = ccl.nl_pt.PTCalculator(with_NC=False,
                                       with_IA=False,
                                       with_dd=False)
    for t in ['TG', 'TI', 'TM']:
        with pytest.raises(ValueError):
            ccl.nl_pt.get_pt_pk2d(COSMO, TRS[t],
                                  nonlin_pk_type='spt',
                                  ptc=ptc_empty)

    # Wrong non-linear Pk
    with pytest.raises(NotImplementedError):
        ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TM'],
                              nonlin_pk_type='halofat')

    # Wrong tracer types
    tdum = ccl.nl_pt.PTMatterTracer()
    tdum.type = 'A'
    for t in ['TG', 'TI', 'TM']:
        with pytest.raises(NotImplementedError):
            ccl.nl_pt.get_pt_pk2d(COSMO, TRS[t],
                                  tracer2=tdum, ptc=PTC)
    with pytest.raises(NotImplementedError):
        ccl.nl_pt.get_pt_pk2d(COSMO, tdum,
                              tracer2=TRS['TM'], ptc=PTC)


def atest_translate_IA_norm():
    a_1 = a_2 = a_d = 1.0
    gz = ccl.growth_factor(COSMO, 1./(1+ZZ))
    ZZ_1 = 1.0
    gz_1 = ccl.growth_factor(COSMO, 1./(1+ZZ_1))
    a_1_v = a_1*np.ones_like(ZZ)
    rho_crit = ccl.physical_constants.RHO_CRITICAL
    Om_m_fid = 0.3

    c_1_t = -1*a_1*5e-14*rho_crit*COSMO['Omega_m']/gz
    c_1_t_1 = -1*a_1*5e-14*rho_crit*COSMO['Omega_m']/gz_1
    c_d_t = -1*a_d*5e-14*rho_crit*COSMO['Omega_m']/gz
    c_2_t = a_2*5*5e-14*rho_crit*COSMO['Omega_m']**2/(Om_m_fid*gz**2)
    c_2_t_des = a_2*5*5e-14*rho_crit*COSMO['Omega_m']/(gz**2)

    # test that it works with scalar a, vector z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, ZZ, a1=a_1, a1delta=a_d,
                                                a2=a_2, Om_m2_for_c2=False)
    assert c_1.all() == c_1_t.all()
    assert c_d.all() == c_d_t.all()
    assert c_2.all() == c_2_t_des.all()

    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, ZZ, a1=a_1, a1delta=a_d,
                                                a2=a_2, Om_m2_for_c2=True)
    assert c_2.all() == c_2_t.all()

    # test that it works with scalar a, scalar z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, ZZ_1, a1=a_1,
                                                Om_m2_for_c2=False)
    assert c_1 == c_1_t_1

    # test that it works with vector a, vector z
    c_1, c_d, c_2 = ccl.nl_pt.translate_IA_norm(COSMO, ZZ, a1=a_1_v,
                                                Om_m2_for_c2=False)
    assert c_1.all() == c_1_t.all()


def atest_return_ptc():
    # if no ptc is input, check that returned pk and ptc objects have
    # the correct properties.
    pk, ptc1 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TG'], return_ptc=True)
    assert isinstance(pk, ccl.Pk2D)
    assert isinstance(ptc1, ccl.nl_pt.PTCalculator)
    # same test with EE/BB output
    pee2, pbb2, ptc2 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'],
                                             return_ia_ee_and_bb=True,
                                             return_ptc=True)
    assert isinstance(pee2, ccl.Pk2D)
    assert isinstance(pbb2, ccl.Pk2D)
    assert isinstance(ptc2, ccl.nl_pt.PTCalculator)
    # check that returned ptc matches input ptc.
    pk_2, ptc1_2 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TG'], ptc=PTC,
                                         return_ptc=True)
    pee2_2, pbb2_2, ptc2_2 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                                                   return_ia_ee_and_bb=True,
                                                   return_ptc=True)
    assert ptc1_2 is PTC
    assert ptc2_2 is PTC
    # check that the result outputs are the same
    # for the internally initialized ptc.
    assert np.allclose(pk_2.eval(ks, 1., COSMO), pk.eval(ks, 1., COSMO))
    assert np.allclose(pee2_2.eval(ks, 1., COSMO), pee2.eval(ks, 1., COSMO))
    assert np.allclose(pbb2_2.eval(ks, 1., COSMO), pbb2.eval(ks, 1., COSMO))


def atest_pt_no_ptc_update():
    pee1 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC)
    pee1_no_update = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                                           update_ptc=False)

    COSMO2 = ccl.Cosmology(
        Omega_c=COSMO["Omega_c"], Omega_b=COSMO["Omega_b"], h=COSMO["h"],
        sigma8=COSMO["sigma8"]+0.05, n_s=COSMO["n_s"],
        transfer_function='bbks', matter_power_spectrum='linear')

    pee2_no_update = ccl.nl_pt.get_pt_pk2d(COSMO2, TRS['TI'], ptc=PTC,
                                           update_ptc=False)
    pee2 = ccl.nl_pt.get_pt_pk2d(COSMO2, TRS['TI'], ptc=PTC)

    assert pee1.eval(0.1, 0.9, COSMO) == pee1_no_update.eval(0.1, 0.9, COSMO)
    assert pee2.eval(0.1, 0.9, COSMO) != pee2_no_update.eval(0.1, 0.9, COSMO2)
