import numpy as np
import pyccl as ccl
import pytest
from contextlib import nullcontext


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
                                       pad_factor=2)
    assert len(c.k_s) == 40


@pytest.mark.parametrize('tr1,tr2,bb,sub_lowk',
                         [['TG', 'TG', False, False],
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
def test_ept_get_pk2d_smoke(tr1, tr2, bb, sub_lowk):
    t2 = None if tr2 == tr1 else TRS[tr2]
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        sub_lowk=sub_lowk, cosmo=COSMO)

    will_warn = set([tr1, tr2]) == set(["TG", "TI"])
    ccl.update_warning_verbosity('high')
    with pytest.warns(ccl.CCLWarning) if will_warn else nullcontext():
        pk = ptc.get_biased_pk2d(TRS[tr1], tracer2=t2, return_ia_bb=bb)
    ccl.update_warning_verbosity('low')
    assert isinstance(pk, ccl.Pk2D)


def test_ept_pk2d_bb_smoke():
    pee = PTC.get_biased_pk2d(TRS['TI'])
    pbb = PTC.get_biased_pk2d(TRS['TI'], return_ia_bb=True)
    assert pee(0.1, 0.9, cosmo=COSMO) != pbb(0.1, 0.9, cosmo=COSMO)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'pt'])
def test_ept_get_pk2d_nl(nl):
    ptc = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        b1_pk_kind=nl, bk2_pk_kind=nl, cosmo=COSMO)
    pk = ptc.get_biased_pk2d(TRS['TG'])
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
    pkmm = ptc1.get_biased_pk2d(tm, tracer2=tm)
    pkmm2 = ptc2.get_biased_pk2d(tm, tracer2=tm)
    pkgg = ptc1.get_biased_pk2d(tg, tracer2=tg)
    ks = np.geomspace(1E-3, 1E1, 128)
    p1 = pkgg(ks, 1., cosmo=COSMO)
    p2 = pkmm(ks, 1., cosmo=COSMO)+ks**2*pkmm2(ks, 1., cosmo=COSMO)
    assert np.allclose(p1, p2, atol=0, rtol=1E-4)


@pytest.mark.parametrize('kind', ccl.nl_pt.ept._PK_ALIAS.keys())
def test_ept_deconstruction(kind):
    ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                         with_matter_1loop=True,
                                         cosmo=COSMO, sub_lowk=True)
    b_nc = ['b1', 'b2', 'b3nl', 'bs', 'bk2']
    b_ia = ['c1', 'c2', 'cdelta']
    pk1 = ptc.get_pk2d_template(kind)

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

    is_nl = tn1 in ["b2", "bs", "bk2", "b3nl"]
    is_g = tn2 in ["c1", "c2", "cdelta"]
    ccl.update_warning_verbosity('high')
    with pytest.warns(ccl.CCLWarning) if is_nl and is_g else nullcontext():
        pk2 = ptc.get_biased_pk2d(t1, tracer2=t2)
    ccl.update_warning_verbosity('low')
    if pk1 is None:
        assert pk2(0.5, 1.0, cosmo=COSMO) == 0.0
    else:
        v1 = pk1(0.5, 1.0, cosmo=COSMO)
        v2 = pk2(0.5, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v2, atol=0, rtol=1e-6)
        # Check cached
        pk3 = ptc._pk2d_temp[ccl.nl_pt.ept._PK_ALIAS[kind]]
        v3 = pk3(0.5, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v3, atol=0, rtol=1e-6)


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

    pk2 = PTC.get_biased_pk2d(t1, tracer2=t2, return_ia_bb=True)

    assert pk1(0.5, 1.0, cosmo=COSMO) == pk2(0.5, 1.0, cosmo=COSMO)


def test_ept_pk_cutoff():
    # Tests the exponential cutoff
    ks = np.geomspace(1E-2, 15., 128)

    t = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                          cosmo=COSMO)
    pk2d1 = ptc1.get_biased_pk2d(t, tracer2=t)
    pk1 = pk2d1(ks, 1.0, cosmo=COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                          k_cutoff=10.,
                                          n_exp_cutoff=2.,
                                          cosmo=COSMO)
    pk2d2 = ptc2.get_biased_pk2d(t, tracer2=t)
    pk2 = pk2d2(ks, 1.0, cosmo=COSMO)

    expcut = np.exp(-(ks/10.)**2)
    assert np.allclose(pk1*expcut, pk2, atol=0, rtol=1E-3)


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
    pk1 = ptc1.get_biased_pk2d(tm)(ks, 1.0, cosmo=COSMO)
    pk2 = ptc2.get_biased_pk2d(tg)(ks, 1.0, cosmo=COSMO)
    assert np.allclose(pk1, pk2, atol=0, rtol=1E-3)


def test_ept_matter_linear():
    # Check that using linear power spectrum as b1 term
    # gets you the SPT power spectrum when asking for the
    # mm power spectrum.
    ks = np.array([0.01, 0.1, 1.0])
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_matter_1loop=True,
                                          b1_pk_kind='linear',
                                          cosmo=COSMO)
    pk1 = ptc1.get_biased_pk2d(TRS['TM'])(ks, 1.0, cosmo=COSMO)
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_matter_1loop=True,
                                          b1_pk_kind='pt',
                                          cosmo=COSMO)
    pk2 = ptc2.get_biased_pk2d(TRS['TM'])(ks, 1.0, cosmo=COSMO)
    assert np.allclose(pk1, pk2, atol=0, rtol=1E-10)


def test_ept_calculator_raises():
    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.EulerianPTCalculator(b1_pk_kind='non-linear')

    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.EulerianPTCalculator(bk2_pk_kind='non-linear')

    # Uninitialized templates
    with pytest.raises(ccl.CCLError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True)
        ptc.get_biased_pk2d(TRS['TG'])

    # Didn't ask for the right calculation
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=False,
                                             cosmo=COSMO)
        ptc.get_biased_pk2d(TRS['TG'])

    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_IA=False,
                                             cosmo=COSMO)
        ptc.get_biased_pk2d(TRS['TI'])

    # Wrong pair combination
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                             cosmo=COSMO)
        ptc.get_pk2d_template('b1:b3')

    # Warning when computing IA-gal correlation
    ccl.update_warning_verbosity('high')
    with pytest.warns(ccl.CCLWarning):
        ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                             cosmo=COSMO)
        tg = ccl.nl_pt.PTNumberCountsTracer(b1=1.0, b2=1.0)
        ptc.get_biased_pk2d(tg, tracer2=TRS['TI'])
    ccl.update_warning_verbosity('low')


def test_ept_template_swap():
    # Test that swapping operator order gets you the same
    # Pk
    ks = np.array([0.01, 0.1, 1.0])
    ptc = ccl.nl_pt.EulerianPTCalculator(with_NC=True,
                                         cosmo=COSMO)
    pk1 = ptc.get_pk2d_template('b2:bs')(ks, 1.0, cosmo=COSMO)
    pk2 = ptc.get_pk2d_template('bs:b2')(ks, 1.0, cosmo=COSMO)
    assert np.all(pk1 == pk2)


def test_ept_eq():
    ptc1 = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                          with_matter_1loop=True)
    # Should be the same
    ptc2 = ccl.nl_pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                                          with_matter_1loop=True)
    assert ptc1 == ptc2
    # Should still be the same
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        a_arr=ccl.pyutils.get_pk_spline_a())
    assert ptc1 == ptc2
    # Different a sampling
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        a_arr=np.linspace(0.5, 1., 30))
    assert ptc1 != ptc2
    # Different PT templates
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=False, with_matter_1loop=True)
    assert ptc1 != ptc2
    # Different FastPT params
    ptc2 = ccl.nl_pt.EulerianPTCalculator(
        with_NC=True, with_IA=True, with_matter_1loop=True,
        C_window=0.5)
    assert ptc1 != ptc2
