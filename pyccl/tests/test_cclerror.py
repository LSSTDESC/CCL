from numpy.testing import assert_
import pyccl


def test_cclerror_repr():
    """Check that a CCLError can be built from its repr"""
    e = pyccl.CCLError("blah")
    e2 = eval(repr(e))
    assert_(str(e2) == str(e))
    assert_(e2 == e)


def test_cclerror_not_equal():
    """Check that a CCLError can be built from its repr"""
    e = pyccl.CCLError("blah")
    e2 = pyccl.CCLError("blahh")
    assert_(e is not e2)
    assert_(e != e2)
    assert_(hash(e) != hash(e2))


def test_cclwarning_repr():
    """Check that a CCLWarning can be built from its repr"""
    w = pyccl.CCLWarning("blah")
    w2 = eval(repr(w))
    assert_(str(w2) == str(w))
    assert_(w2 == w)

    v = pyccl.CCLDeprecationWarning("blah")
    v2 = eval(repr(v))
    assert str(v2) == str(v)
    assert v2 == v


def test_cclwarning_not_equal():
    """Check that a CCLWarning can be built from its repr"""
    w = pyccl.CCLWarning("blah")
    w2 = pyccl.CCLWarning("blahh")
    assert_(w is not w2)
    assert_(w != w2)
    assert_(hash(w) != hash(w2))

    v = pyccl.CCLDeprecationWarning("blah")
    v2 = pyccl.CCLDeprecationWarning("blahh")
    assert v is not v2
    assert v != v2
    assert hash(v) != hash(v2)


def test_ccl_deprecationwarning_switch():
    import warnings

    # check that warnings are enabled by default
    with warnings.catch_warnings(record=True) as w:
        warnings.warn("test", pyccl.CCLDeprecationWarning)
    assert w[0].category == pyccl.CCLDeprecationWarning

    # switch off CCL (future) deprecation warnings
    pyccl.CCLDeprecationWarning.disable()
    with warnings.catch_warnings(record=True) as w:
        warnings.warn("test", pyccl.CCLDeprecationWarning)
    assert len(w) == 0

    # switch back on
    pyccl.CCLDeprecationWarning.enable()
