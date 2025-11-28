"""Tests for importing the pyccl module."""

def test_import_pyccl():
    import pyccl

def test_import_has_symbols():
    import pyccl
    assert hasattr(pyccl, "lib")
