import numpy as np
import pyccl as ccl
import pytest

NZ = 128
ZZ = np.linspace(0., 1., NZ)
BZ_C = 2.
BZ = BZ_C * np.ones(NZ)


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
    ccl.PTWorkspace()
