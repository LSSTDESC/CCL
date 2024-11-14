import pyccl as ccl


def test_version():
    # Check that dynamic versioning is working
    assert ccl.__version__ != '0.0.0'


def test_debug_mode_toggle():
    ccl.debug_mode(True)
    ccl.debug_mode(False)
