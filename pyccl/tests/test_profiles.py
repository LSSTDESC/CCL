import pytest
import pyccl as ccl


def test_profile_defaults():
    p = ccl.halos.HaloProfile()
    with pytest.raises(NotImplementedError):
        p.profile_real(None, None, None, None)
    with pytest.raises(NotImplementedError):
        p.profile_fourier(None, None, None, None)
