import pyccl as ccl
import pytest


def test_unexpected_argument_raises():
    # Test that if an argument has been renamed it will still raise when
    # a wrong argument is passed.
    with pytest.raises(TypeError):
        # here, `c_m` was renamed to `concentration`
        ccl.halos.MassDef200c(hello="Duffy08")
