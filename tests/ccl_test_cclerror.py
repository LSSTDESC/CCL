from __future__ import print_function
from numpy.testing import run_module_suite, assert_
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


if __name__ == '__main__':
    run_module_suite()
