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


def test_cclwarning_not_equal():
    """Check that a CCLWarning can be built from its repr"""
    w = pyccl.CCLWarning("blah")
    w2 = pyccl.CCLWarning("blahh")
    assert_(w is not w2)
    assert_(w != w2)
    assert_(hash(w) != hash(w2))
