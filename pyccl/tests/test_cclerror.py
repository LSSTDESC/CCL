import pyccl


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
