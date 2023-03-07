import pyccl as ccl


def test_debug_mode_toggle():
    ccl.debug_mode(True)
    ccl.debug_mode(False)
