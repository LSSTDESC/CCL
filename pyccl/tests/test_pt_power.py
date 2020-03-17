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
PTC = ccl.nl_pt.PTCalculator(with_NC=True,
                             with_IA=True,
                             with_dd=True)


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


def test_pt_workspace_smoke():
    w = ccl.nl_pt.PTCalculator(log10k_min=-3,
                               log10k_max=1,
                               nk_per_decade=10,
                               pad_factor=2)
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
    pk = ccl.nl_pt.get_pt_pk2d(COSMO,
                               TRS[options[0]],
                               tracer2=t2,
                               return_ia_bb=options[2],
                               sub_lowk=options[3],
                               ptc=options[4])
    assert isinstance(pk, ccl.Pk2D)


def test_pt_pk2d_bb():
    pee = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC)
    pbb = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                                return_ia_bb=True)
    pee2, pbb2 = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TI'], ptc=PTC,
                                       return_ia_ee_and_bb=True)
    assert pee.eval(0.1, 0.9, COSMO) == pee2.eval(0.1, 0.9, COSMO)
    assert pbb.eval(0.1, 0.9, COSMO) == pbb2.eval(0.1, 0.9, COSMO)


@pytest.mark.parametrize('nl', ['nonlinear', 'linear', 'spt'])
def test_pt_get_pk2d_nl(nl):
    pk = ccl.nl_pt.get_pt_pk2d(COSMO, TRS['TG'],
                               nonlin_pk_type=nl)
    assert isinstance(pk, ccl.Pk2D)


def test_ptc_raises():
    with pytest.raises(ValueError):
        PTC.update_pk(np.zeros(4))


def test_pt_get_pk2d_raises():
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
