import numpy as np
import pyccl as ccl
import pytest


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
TRS = {'TG': ccl.nl_pt.PTNumberCountsTracer(b1=2.0, b2=2.0, bs=2.0),
       'TI': ccl.nl_pt.PTIntrinsicAlignmentTracer(c1=2.0, c2=2.0,
                                                  cdelta=2.0),
       'TM': ccl.nl_pt.PTMatterTracer()}
PTC = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                     with_matter_1loop=True,
                                     cosmo=COSMO)


def test_ept_calculator_smoke():
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
def test_ept_get_pk2d_smoke(options):
    if options[0] == options[1]:
        t2 = None
    else:
        t2 = TRS[options[1]]
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        extra_params={'sub_lowk': options[3]},
        cosmo=COSMO)
    pk = ptc.get_pk2d_biased(TRS[options[0]],
                             tracer2=t2,
                             return_ia_bb=options[2])
    assert isinstance(pk, ccl.Pk2D)


def test_ept_pk2d_bb_smoke():
    pee = PTC.get_pk2d_biased(TRS['TI'])
    pbb = PTC.get_pk2d_biased(TRS['TI'], return_ia_bb=True)
    assert pee.eval(0.1, 0.9, COSMO) != pbb.eval(0.1, 0.9, COSMO)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'pt'])
def test_ept_get_pk2d_nl(nl):
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=nl, bk2_pk_kind=nl, cosmo=COSMO)
    pk = ptc.get_pk2d_biased(TRS['TG'])
    assert isinstance(pk, ccl.Pk2D)


@pytest.mark.parametrize('typ_nlin,typ_nloc', [('linear', 'nonlinear'),
                                               ('nonlinear', 'linear'),
                                               ('nonlinear', 'pt'),
                                               ('linear', 'pt')])
def test_ept_k2pk_types(typ_nlin, typ_nloc):
    tg = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0., bk2=1.)
    tm = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0.)
    ptc1 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=typ_nlin, bk2_pk_kind=typ_nloc,
        cosmo=COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=typ_nloc, cosmo=COSMO)
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
def test_ept_deconstruction(kind):
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
def test_ept_deconstruction_bb(kind):
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


def test_ept_pk_cutoff():
    # Tests the exponential cutoff
    ks = np.geomspace(1E-2, 15., 128)

    t = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                          cosmo=COSMO)
    pk2d1 = ptc1.get_pk2d_biased(t, tracer2=t)
    pk1 = pk2d1.eval(ks, 1.0, COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                          k_cutoff=10.,
                                          n_exp_cutoff=2.,
                                          cosmo=COSMO)
    pk2d2 = ptc2.get_pk2d_biased(t, tracer2=t)
    pk2 = pk2d2.eval(ks, 1.0, COSMO)

    expcut = np.exp(-(ks/10.)**2)
    assert np.all(np.fabs(pk1*expcut/pk2-1) < 1E-3)


def test_ept_matter_1loop():
    # Check P(k) for linear tracer with b1=1 is the same
    # as matter P(k)
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_NC=False, with_IA=False,
                                          with_matter_1loop=True,
                                          cosmo=COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_NC=True, cosmo=COSMO)
    tg = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    tm = ccl.nl_pt.PTMatterTracer()
    ks = np.geomspace(1E-3, 10, 64)
    pk1 = ptc1.get_pk2d_biased(tm).eval(ks, 1.0, COSMO)
    pk2 = ptc2.get_pk2d_biased(tg).eval(ks, 1.0, COSMO)
    assert np.all(np.fabs(pk1/pk2-1) < 1E-3)


def test_ept_matter_linear():
    # Check that using linear power spectrum as b1 term
    # gets you the SPT power spectrum when asking for the
    # mm power spectrum.
    ks = np.array([0.01, 0.1, 1.0])
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_matter_1loop=True,
                                          b1_pk_kind='linear',
                                          cosmo=COSMO)
    pk1 = ptc1.get_pk2d_biased(TRS['TM']).eval(ks, 1.0, COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_matter_1loop=True,
                                          b1_pk_kind='pt',
                                          cosmo=COSMO)
    pk2 = ptc2.get_pk2d_biased(TRS['TM']).eval(ks, 1.0, COSMO)
    assert np.all(np.fabs(pk1/pk2-1) < 1E-10)


def test_ept_calculator_raises():
    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.EulerianPTCalculator(b1_pk_kind='non-linear')

    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.EulerianPTCalculator(bk2_pk_kind='non-linear')

    # Uninitialized templates
    with pytest.raises(RuntimeError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True)
        ptc.get_pk2d_biased(TRS['TG'])

    # Didn't ask for the right calculation
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=False,
                                             cosmo=COSMO)
        ptc.get_pk2d_biased(TRS['TG'])

    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_IA=False,
                                             cosmo=COSMO)
        ptc.get_pk2d_biased(TRS['TI'])

    # Wrong pair combination
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                             cosmo=COSMO)
        ptc.get_pk2d_template('b1:b3')

    # Warning when computing IA-gal correlation
    with pytest.warns(ccl.CCLWarning):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                             cosmo=COSMO)
        tg = ccl.nl_pt.PTNumberCountsTracer(b1=1.0, b2=1.0)
        ptc.get_pk2d_biased(tg, tracer2=TRS['TI'])


def test_ept_template_swap():
    # Test that swapping operator order gets you the same
    # Pk
    ks = np.array([0.01, 0.1, 1.0])
    ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                         cosmo=COSMO)
    pk1 = ptc.get_pk2d_template('b2:bs').eval(ks, 1.0, COSMO)
    pk2 = ptc.get_pk2d_template('bs:b2').eval(ks, 1.0, COSMO)
    assert np.all(pk1 == pk2)
