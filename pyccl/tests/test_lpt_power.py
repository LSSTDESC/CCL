import numpy as np
import pyccl as ccl
import pytest


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
TRS = {'TG': ccl.nl_pt.PTNumberCountsTracer(b1=2.0, b2=2.0, bs=2.0, bk2=2.0),
       'TM': ccl.nl_pt.PTMatterTracer()}
PTC = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)


def test_lpt_calculator_smoke():
    c = ccl.nl_pt.LagrangianPTCalculator(log10k_min=-3,
                                         log10k_max=1,
                                         nk_per_decade=10)
    assert len(c.k_s) == 40


@pytest.mark.parametrize('tr1,tr2',
                         [['TG', 'TG'],
                          ['TG', 'TM'],
                          ['TM', 'TG'],
                          ['TM', 'TM']])
def test_lpt_get_pk2d_smoke(tr1, tr2):
    t2 = None if tr2 == tr1 else TRS[tr2]
    ptc = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)

    pk = ptc.get_biased_pk2d(TRS[tr1], tracer2=t2)
    assert isinstance(pk, ccl.Pk2D)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'pt'])
def test_lpt_get_pk2d_nl(nl):
    ptc = ccl.nl_pt.LagrangianPTCalculator(
        b1_pk_kind=nl, bk2_pk_kind=nl, cosmo=COSMO)
    pk = ptc.get_biased_pk2d(TRS['TG'])
    assert isinstance(pk, ccl.Pk2D)


@pytest.mark.parametrize('typ_nlin,typ_nloc', [('linear', 'nonlinear'),
                                               ('nonlinear', 'linear')])
def test_lpt_k2pk_types(typ_nlin, typ_nloc):
    tg = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0., bk2=1.)
    tm = ccl.nl_pt.PTNumberCountsTracer(1., 0., 0.)
    ptc1 = ccl.nl_pt.LagrangianPTCalculator(
        b1_pk_kind=typ_nlin, bk2_pk_kind=typ_nloc, cosmo=COSMO)
    ptc2 = ccl.nl_pt.LagrangianPTCalculator(
        b1_pk_kind=typ_nloc, cosmo=COSMO)
    pkmm = ptc1.get_biased_pk2d(tm, tracer2=tm)
    pkmm2 = ptc2.get_biased_pk2d(tm, tracer2=tm)
    pkgg = ptc1.get_biased_pk2d(tg, tracer2=tg)
    ks = np.geomspace(1E-3, 1E1, 128)
    p1 = pkgg(ks, 1., cosmo=COSMO)
    p2 = pkmm(ks, 1., cosmo=COSMO)+ks**2*pkmm2(ks, 1., cosmo=COSMO)
    assert np.allclose(p1, p2, atol=0, rtol=1E-4)


@pytest.mark.parametrize('kind', ['m:m', 'm:b2', 'm:b3nl', 'm:bs',
                                  'b1:b3nl', 'b3nl:b3nl', 'b3nl:bk2'])
def test_lpt_deconstruction(kind):
    ptc = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO,
                                           b1_pk_kind='nonlinear',
                                           bk2_pk_kind='nonlinear')
    b_nc = ['b1', 'b2', 'b3nl', 'bs', 'bk2']
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

    tn1, tn2 = kind.split(':')
    t1 = get_tr(tn1)
    t2 = get_tr(tn2)

    pk2 = ptc.get_biased_pk2d(t1, tracer2=t2)
    if pk1 is None:
        assert pk2(0.5, 1.0, cosmo=COSMO) == 0.0
    else:
        v1 = pk1(0.5, 1.0, cosmo=COSMO)
        v2 = pk2(0.5, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v2, atol=0, rtol=1e-6)
        # Check cached
        pk3 = ptc._pk2d_temp[ccl.nl_pt.lpt._PK_ALIAS[kind]]
        v3 = pk3(0.5, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v3, atol=0, rtol=1e-6)


def test_lpt_pk_cutoff():
    # Tests the exponential cutoff
    ks = np.geomspace(1E-2, 15., 128)

    t = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    ptc1 = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)
    pk2d1 = ptc1.get_biased_pk2d(t, tracer2=t)
    pk1 = pk2d1(ks, 1.0, cosmo=COSMO)
    ptc2 = ccl.nl_pt.LagrangianPTCalculator(k_cutoff=10.,
                                            n_exp_cutoff=2.,
                                            cosmo=COSMO)
    pk2d2 = ptc2.get_biased_pk2d(t, tracer2=t)
    pk2 = pk2d2(ks, 1.0, cosmo=COSMO)

    expcut = np.exp(-(ks/10.)**2)
    assert np.allclose(pk1*expcut, pk2, atol=0, rtol=1E-3)


def test_lpt_matter_1loop():
    # Check P(k) for linear tracer with b1=1 is the same
    # as matter P(k)
    ptc = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO,
                                           b1_pk_kind='pt')
    tg = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    tm = ccl.nl_pt.PTMatterTracer()
    ks = np.geomspace(1E-3, 10, 64)
    pk1 = ptc.get_biased_pk2d(tm)(ks, 1.0, cosmo=COSMO)
    pk2 = ptc.get_biased_pk2d(tg)(ks, 1.0, cosmo=COSMO)
    assert np.allclose(pk1, pk2, atol=0, rtol=1E-3)


def test_lpt_calculator_raises():
    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.LagrangianPTCalculator(b1_pk_kind='non-linear')

    # Wrong type of b1 and bk2 power spectra
    with pytest.raises(ValueError):
        ccl.nl_pt.LagrangianPTCalculator(bk2_pk_kind='non-linear')

    # Uninitialized templates
    with pytest.raises(ccl.CCLError):
        ptc = ccl.nl_pt.LagrangianPTCalculator()
        ptc.get_biased_pk2d(TRS['TG'])

    # TODO: Discuss this test
    # Wrong pair combination
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)
        ptc.get_pk2d_template('b1:b3')


def test_lpt_template_swap():
    # Test that swapping operator order gets you the same
    # Pk
    ks = np.array([0.01, 0.1, 1.0])
    ptc = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)
    pk1 = ptc.get_pk2d_template('b2:bs')(ks, 1.0, cosmo=COSMO)
    pk2 = ptc.get_pk2d_template('bs:b2')(ks, 1.0, cosmo=COSMO)
    assert np.all(pk1 == pk2)


def test_lpt_eq():
    ptc1 = ccl.nl_pt.LagrangianPTCalculator()
    # Should be the same
    ptc2 = ccl.nl_pt.LagrangianPTCalculator()
    assert ptc1 == ptc2
    # Should still be the same
    ptc2 = ccl.nl_pt.LagrangianPTCalculator(
        a_arr=ccl.pyutils.get_pk_spline_a())
    assert ptc1 == ptc2
    # Different a sampling
    ptc2 = ccl.nl_pt.LagrangianPTCalculator(
        a_arr=np.linspace(0.5, 1., 30))
    assert ptc1 != ptc2
