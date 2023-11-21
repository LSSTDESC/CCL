import numpy as np
import pyccl as ccl
import pytest


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
TRS = {'TG': ccl.nl_pt.PTNumberCountsTracer(b1=2.0, b2=2.0, bs=2.0, bk2=2.0),
       'TM': ccl.nl_pt.PTMatterTracer()}
PTC = ccl.nl_pt.LagrangianPTCalculator(cosmo=COSMO)


def test_bacco_lbias_calculator_smoke():
    c = ccl.nl_pt.BaccoLbiasCalculator(log10k_min=-3,
                                       log10k_max=1,
                                       nk_per_decade=10)
    assert len(c.k_s) == 40


@pytest.mark.parametrize('tr1,tr2',
                         [['TG', 'TG'],
                          ['TG', 'TM'],
                          ['TM', 'TG'],
                          ['TM', 'TM']])
def test_bacco_lbias_get_pk2d_smoke(tr1, tr2):
    t2 = None if tr2 == tr1 else TRS[tr2]
    ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)

    pk = ptc.get_biased_pk2d(TRS[tr1], tracer2=t2)
    assert isinstance(pk, ccl.Pk2D)


def test_bacco_lbias_get_pk2d_nl():
    ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
    pk = ptc.get_biased_pk2d(TRS['TG'])
    assert isinstance(pk, ccl.Pk2D)


@pytest.mark.parametrize('typ_nlin,typ_nloc', [('linear', 'nonlinear'),
                                               ('nonlinear', 'linear')])
def test_bacco_lbias_k2pk_types(typ_nlin, typ_nloc):
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


@pytest.mark.parametrize('kind', ['m:m', 'm:b1', 'm:b2', 'm:bs', 'm:bk2',
                                  'b1:b1', 'b1:b2', 'b1:bs', 'b1:bk2', 'b2:b2',
                                  'b2:bs', 'b2:bk2', 'bs:bs', 'bs:bk2',
                                  'bk2:bk2', 'b1:b3nl'])
def test_bacco_lbias_deconstruction(kind):
    ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
    b_nc = ['b1', 'b2', 'bs', 'bk2', 'b3nl']
    pk1 = ptc.get_pk2d_template(kind)

    def get_tr(tn):
        if tn == 'm':
            return ccl.nl_pt.PTMatterTracer()
        if tn in b_nc:
            bdict = {b: 0.0 for b in b_nc}
            # This is b1E = 1, so that internally it goes to b1L = 0
            if tn == 'b1':
                bdict[tn] = 2.0
            else:
                bdict['b1'] = 1.0
                bdict[tn] = 1.0
            return ccl.nl_pt.PTNumberCountsTracer(
                b1=bdict['b1'], b2=bdict['b2'],
                bs=bdict['bs'], bk2=bdict['bk2'],
                b3nl=bdict['b3nl'])

    tn1, tn2 = kind.split(':')
    t1 = get_tr(tn1)
    t2 = get_tr(tn2)

    pkmm = ptc.get_pk2d_template('m:m')
    pkx1 = ptc.get_pk2d_template(f'm:{tn1}')
    pkx2 = ptc.get_pk2d_template(f'm:{tn2}')

    pk2 = ptc.get_biased_pk2d(t1, tracer2=t2)
    if pk1 is None:
        assert np.allclose(pk2(0.2, 1.0, cosmo=COSMO),
                           pkmm(0.2, 1.0, cosmo=COSMO)
                           + pkx1(0.2, 1.0, cosmo=COSMO),
                           atol=0, rtol=1e-6)
    else:
        if (t1.type == 'M') & (t2.type == 'M'):
            v1 = pk1(0.2, 1.0, cosmo=COSMO)
        elif (t1.type == 'M') | (t2.type == 'M'):
            v1 = pkmm(0.2, 1.0, cosmo=COSMO) + pk1(0.2, 1.0, cosmo=COSMO)
        else:
            v1 = pkmm(0.2, 1.0, cosmo=COSMO) + pkx1(0.2, 1.0, cosmo=COSMO) \
                + pkx2(0.2, 1.0, cosmo=COSMO) + pk1(0.2, 1.0, cosmo=COSMO)
        v2 = pk2(0.2, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v2, atol=0, rtol=1e-6)
        # Check cached
        pk3 = ptc._pk2d_temp[ccl.nl_pt.lpt._PK_ALIAS[kind]]
        chached_pkmm = ptc._pk2d_temp[ccl.nl_pt.lpt._PK_ALIAS['m:m']]
        chached_pkx1 = ptc._pk2d_temp[ccl.nl_pt.lpt._PK_ALIAS[f'm:{tn1}']]
        chached_pkx2 = ptc._pk2d_temp[ccl.nl_pt.lpt._PK_ALIAS[f'm:{tn2}']]
        if (t1.type == 'M') & (t2.type == 'M'):
            v3 = pk3(0.2, 1.0, cosmo=COSMO)
        elif (t1.type == 'M') | (t2.type == 'M'):
            v3 = chached_pkmm(0.2, 1.0, cosmo=COSMO) \
                + pk3(0.2, 1.0, cosmo=COSMO)
        else:
            v3 = chached_pkmm(0.2, 1.0, cosmo=COSMO) \
                + chached_pkx1(0.2, 1.0, cosmo=COSMO) \
                + chached_pkx2(0.2, 1.0, cosmo=COSMO) \
                + pk3(0.2, 1.0, cosmo=COSMO)
        assert np.allclose(v1, v3, atol=0, rtol=1e-6)


def test_bacco_lbias_pk_cutoff():
    # Tests the exponential cutoff
    ks = np.geomspace(1E-2, 0.3, 30)

    t = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    ptc1 = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
    pk2d1 = ptc1.get_biased_pk2d(t, tracer2=t)
    pk1 = pk2d1(ks, 1.0, cosmo=COSMO)
    ptc2 = ccl.nl_pt.BaccoLbiasCalculator(k_cutoff=0.1,
                                          n_exp_cutoff=2.,
                                          cosmo=COSMO)
    pk2d2 = ptc2.get_biased_pk2d(t, tracer2=t)
    pk2 = pk2d2(ks, 1.0, cosmo=COSMO)

    expcut = np.exp(-(ks/0.1)**2)
    assert np.allclose(pk1*expcut, pk2, atol=0, rtol=1E-2)


def test_bacco_lbias_matter_1loop():
    # Check P(k) for linear tracer with b1=1 is the same
    # as matter P(k)
    ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
    tg = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)
    tm = ccl.nl_pt.PTMatterTracer()
    ks = np.geomspace(1E-3, 10, 64)
    pk1 = ptc.get_biased_pk2d(tm)(ks, 1.0, cosmo=COSMO)
    pk2 = ptc.get_biased_pk2d(tg)(ks, 1.0, cosmo=COSMO)
    assert np.allclose(pk1, pk2, atol=0, rtol=1E-3)


def test_bacco_lbias_calculator_raises():
    # Uninitialized templates
    with pytest.raises(ccl.CCLError):
        ptc = ccl.nl_pt.BaccoLbiasCalculator()
        ptc.get_biased_pk2d(TRS['TG'])

    # TODO: Discuss this test
    # Wrong pair combination
    with pytest.raises(ValueError):
        ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
        ptc.get_pk2d_template('b1:b3')

    with pytest.raises(ValueError):
        t1 = ccl.nl_pt.PTIntrinsicAlignmentTracer(c1=1)
        ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
        ptc.get_biased_pk2d(t1, tracer2=t1)


def test_bacco_lbias_template_swap():
    # Test that swapping operator order gets you the same
    # Pk
    ks = np.array([0.01, 0.1, 1.0])
    ptc = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO)
    pk1 = ptc.get_pk2d_template('b2:bs')(ks, 1.0, cosmo=COSMO)
    pk2 = ptc.get_pk2d_template('bs:b2')(ks, 1.0, cosmo=COSMO)
    assert np.all(pk1 == pk2)


def test_bacco_lbias_eq():
    ptc1 = ccl.nl_pt.BaccoLbiasCalculator()
    # Should be the same
    ptc2 = ccl.nl_pt.BaccoLbiasCalculator()
    assert ptc1 == ptc2
    # Should still be the same
    ptc2 = ccl.nl_pt.BaccoLbiasCalculator(
        a_arr=ccl.pyutils.get_pk_spline_a())
    assert ptc1 == ptc2
    # Different a sampling
    ptc2 = ccl.nl_pt.BaccoLbiasCalculator(
        a_arr=np.linspace(0.5, 1., 30))
    assert ptc1 != ptc2
    # Should do nothing if cosmo is the same
    ptc2 = ccl.nl_pt.BaccoLbiasCalculator(
        cosmo=COSMO)
    lpt_table_1 = ptc2.lpt_table
    ptc2.update_ingredients(COSMO)
    assert lpt_table_1 is ptc2.lpt_table


def test_bacco_lbias_sigma8_A_s():
    ks = np.logspace(-2, np.log10(0.3), 30)

    t = ccl.nl_pt.PTNumberCountsTracer(b1=1.0)

    COSMO_A_s = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, n_s=0.96,
                              A_s=2.1265e-9, m_nu=0.2)
    COSMO_sigma8 = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, n_s=0.96,
                                 sigma8=0.8059043572377348, m_nu=0.2)

    ptc1 = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO_A_s)
    pk2d1 = ptc1.get_biased_pk2d(t, tracer2=t)
    pk1 = pk2d1(ks, 1.0, cosmo=COSMO_A_s)

    ptc2 = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO_sigma8)
    pk2d2 = ptc2.get_biased_pk2d(t, tracer2=t)
    pk2 = pk2d2(ks, 1.0, cosmo=COSMO_sigma8)

    assert np.allclose(pk1, pk2, atol=0, rtol=1E-3)


def test_bacco_lbias_many_A_s():
    COSMO_sigma8 = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, n_s=0.96,
                                 sigma8=0.81, m_nu=0.2)
    pt = ccl.nl_pt.BaccoLbiasCalculator(cosmo=COSMO_sigma8)
    emupars = dict(
        omega_cold=COSMO_sigma8['Omega_c'] + COSMO_sigma8['Omega_b'],
        omega_baryon=COSMO_sigma8['Omega_b'],
        ns=COSMO_sigma8['n_s'],
        hubble=COSMO_sigma8['h'],
        neutrino_mass=np.sum(COSMO_sigma8['m_nu']),
        w0=COSMO_sigma8['w0'],
        wa=COSMO_sigma8['wa'],
        expfactor=pt.a_s
    )
    sigma8tot = COSMO_sigma8['sigma8']
    sigma8cold = pt._sigma8tot_2_sigma8cold(emupars, sigma8tot)
    newemupars = {}
    for key in emupars:
        newemupars[key] = np.full(len(pt.a_s), emupars[key])
    sigma8cold_arr = pt._sigma8tot_2_sigma8cold(newemupars, sigma8tot)
    assert np.allclose(np.mean(sigma8cold_arr), sigma8cold, atol=0, rtol=1E-4)
