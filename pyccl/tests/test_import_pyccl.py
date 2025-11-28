"""Unit tests for importing the pyccl module."""


def test_import_pyccl_smoke():
    """Tests that pyccl can be imported and its C library is accessible."""
    import pyccl

    assert hasattr(pyccl, "lib")
    assert pyccl.lib is not None
