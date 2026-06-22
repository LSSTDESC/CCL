import pyccl
import pytest
import numpy as np
import warnings


def test_cclerror_repr():
    """Check that a CCLError can be built from its repr"""
    e = pyccl.CCLError("blah")
    e2 = eval(repr(e))
    assert str(e2) == str(e)
    assert e2 == e


def test_cclerror_not_equal():
    """Check that a CCLError can be built from its repr"""
    e = pyccl.CCLError("blah")
    e2 = pyccl.CCLError("blahh")
    assert e is not e2
    assert e != e2
    assert hash(e) != hash(e2)


def test_cclwarning_repr():
    """Check that a CCLWarning can be built from its repr"""
    w = pyccl.CCLWarning("blah")
    w2 = eval(repr(w))
    assert str(w2) == str(w)
    assert w2 == w

    v = pyccl.CCLDeprecationWarning("blah")
    v2 = eval(repr(v))
    assert str(v2) == str(v)
    assert v2 == v


def test_cclwarning_not_equal():
    """Check that a CCLWarning can be built from its repr"""
    w = pyccl.CCLWarning("blah")
    w2 = pyccl.CCLWarning("blahh")
    assert w is not w2
    assert w != w2
    assert hash(w) != hash(w2)

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


def test_ccl_warning_verbosity_error():
    with pytest.raises(KeyError):
        pyccl.update_warning_verbosity("hihg")


def test_ccl_warning_verbosity():

    # The code below will trigger an unimportant warning
    # about the N(z) sampling

    # Switch to high verbosity
    pyccl.update_warning_verbosity("high")
    cosmo = pyccl.CosmologyVanillaLCDM()
    numz = 32
    zm = 0.7
    sz = 0.01
    z = np.linspace(0, 1.5, numz)
    nz = np.exp(-0.5*((z-zm)/sz)**2)
    with pytest.warns(pyccl.CCLWarning):
        pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))

    # Now test that no warning is triggered if back to low verbosity
    pyccl.update_warning_verbosity("low")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyccl.WeakLensingTracer(cosmo, dndz=(z, nz))


def test_ccl_deprecation_warning():
    # Switch to high verbosity to catch it
    pyccl.update_warning_verbosity("high")
    with pytest.warns(pyccl.CCLDeprecationWarning):
        pyccl.baryons.BaccoemuBaryons()
    pyccl.update_warning_verbosity("low")
