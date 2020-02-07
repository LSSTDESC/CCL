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
TRS = {'TG': ccl.PTNumberCountsTracer((ZZ, BZ),
                                      (ZZ, BZ),
                                      (ZZ, BZ)),
       'TI': ccl.PTIntrinsicAlignmentTracer((ZZ, BZ),
                                            (ZZ, BZ),
                                            (ZZ, BZ)),
       'TM': ccl.PTMatterTracer()}
WW = ccl.PTWorkspace()


def test_pt_tracer_smoke():
    ccl.PTTracer()


def test_pt_tracer_m_smoke():
    ccl.PTMatterTracer()


@pytest.mark.parametrize('b2', [(ZZ, BZ), BZ_C, None])
def test_pt_tracer_nc_smoke(b2):
    pt_tr = ccl.PTNumberCountsTracer((ZZ, BZ),
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
    pt_tr = ccl.PTIntrinsicAlignmentTracer((ZZ, BZ),
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


def test_pt_workspace_smoke():
    w = ccl.PTWorkspace(log10k_min=-3,
                        log10k_max=1,
                        nk_per_decade=10,
                        pad_factor=2)
    assert len(w.ks) == 40


@pytest.mark.parametrize('tracers', [['TG', 'TG', False],
                                     ['TG', 'TI', False],
                                     ['TG', 'TM', False],
                                     ['TI', 'TG', False],
                                     ['TI', 'TI', False],
                                     ['TI', 'TI', True],
                                     ['TI', 'TM', False],
                                     ['TM', 'TG', False],
                                     ['TM', 'TI', False],
                                     ['TM', 'TM', False]])
def test_pt_get_pk2d_smoke(tracers):
    ccl.get_pt_pk2d(COSMO, WW,
                    TRS[tracers[0]],
                    TRS[tracers[1]])
