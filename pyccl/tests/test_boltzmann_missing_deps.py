"""Tests that verify helpful CCLError messages when boltzmann optional
dependencies (camb, classy, isitgr) are not installed.

These tests use monkeypatching to simulate missing packages so they do not
require camb or classy to be installed.
"""

import builtins
import sys

import numpy as np
import pytest

import pyccl as ccl


def _make_import_blocker(*blocked_roots):
    """Return a replacement for builtins.__import__ that raises
    ImportError for any module whose root name is in *blocked_roots*.
    Using ImportError (parent of ModuleNotFoundError) mirrors the behaviour of
    ``sys.modules[name] = None`` and covers both flavours of import failure.
    """
    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in blocked_roots:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    return _mock_import


@pytest.fixture()
def _block_camb(monkeypatch):
    """Remove camb from sys.modules and block all subsequent imports of it."""
    for key in list(sys.modules):
        if key == "camb" or key.startswith("camb."):
            monkeypatch.delitem(sys.modules, key)
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("camb"))


@pytest.fixture()
def _block_classy(monkeypatch):
    """Remove classy from sys.modules and block all subsequent imports."""
    monkeypatch.delitem(sys.modules, "classy", raising=False)
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("classy"))


@pytest.fixture()
def _block_isitgr(monkeypatch):
    """Remove isitgr from sys.modules and block all subsequent imports."""
    for key in list(sys.modules):
        if key == "isitgr" or key.startswith("isitgr."):
            monkeypatch.delitem(sys.modules, key)
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("isitgr"))


# ---------------------------------------------------------------------------
# Direct function tests
# ---------------------------------------------------------------------------

def test_camb_missing_raises_cclerror(_block_camb):
    """get_camb_pk_lin raises CCLError (not bare ModuleNotFoundError) with
    an actionable install message when camb is not available."""
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_camb",
    )
    with pytest.raises(ccl.CCLError, match="CAMB"):
        cosmo.get_camb_pk_lin()


def test_camb_missing_error_has_install_hint(_block_camb):
    """Error message contains pip install hint."""
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_camb",
    )
    with pytest.raises(ccl.CCLError, match="pip install"):
        cosmo.get_camb_pk_lin()


def test_class_missing_raises_cclerror(_block_classy):
    """get_class_pk_lin raises CCLError when classy is not available."""
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_class",
    )
    with pytest.raises(ccl.CCLError, match="CLASS|classy"):
        cosmo.get_class_pk_lin()


def test_class_missing_error_has_install_hint(_block_classy):
    """CLASS error message contains pip install hint."""
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_class",
    )
    with pytest.raises(ccl.CCLError, match="pip install"):
        cosmo.get_class_pk_lin()


def test_isitgr_missing_raises_cclerror(_block_isitgr):
    """get_isitgr_pk_lin raises CCLError when isitgr is not available."""
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_isitgr",
    )
    with pytest.raises(ccl.CCLError, match="ISiTGR"):
        cosmo.get_isitgr_pk_lin()


# ---------------------------------------------------------------------------
# Full pipeline tests (reproduces the scenario in test.py)
# ---------------------------------------------------------------------------

def test_halo_massfunction_camb_missing_raises_cclerror(_block_camb):
    """Calling a halo mass function with a boltzmann_camb cosmology raises
    CCLError (not bare ModuleNotFoundError) when camb is not installed.

    This replicates the exact scenario in test.py.
    """
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_camb",
    )
    hmd = ccl.halos.MassDef200m
    nM = ccl.halos.MassFuncTinker08(mass_def=hmd)
    m_arr = np.geomspace(1e8, 1e17, 16)

    with pytest.raises(ccl.CCLError, match="CAMB"):
        nM(cosmo, m_arr, 1.0)


def test_halo_massfunction_class_missing_raises_cclerror(_block_classy):
    """Calling a halo mass function with a boltzmann_class cosmology raises
    CCLError (not bare ModuleNotFoundError) when classy is not installed.
    """
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_class",
    )
    hmd = ccl.halos.MassDef200m
    nM = ccl.halos.MassFuncTinker08(mass_def=hmd)
    m_arr = np.geomspace(1e8, 1e17, 16)

    with pytest.raises(ccl.CCLError, match="CLASS|classy"):
        nM(cosmo, m_arr, 1.0)


def test_error_is_cclerror_not_module_not_found_error(_block_camb):
    """Verify the exception type is CCLError, not ModuleNotFoundError.

    Before this fix, users would get an unhelpful ModuleNotFoundError deep
    in the call stack. The fix ensures it is always a CCLError.
    """
    cosmo = ccl.Cosmology(
        Omega_b=0.0492, Omega_c=0.2650, h=0.6724,
        sigma8=0.811, n_s=0.9645,
        transfer_function="boltzmann_camb",
    )
    hmd = ccl.halos.MassDef200m
    nM = ccl.halos.MassFuncTinker08(mass_def=hmd)
    m_arr = np.geomspace(1e8, 1e17, 16)

    with pytest.raises(ccl.CCLError):
        nM(cosmo, m_arr, 1.0)

    # Also confirm it is NOT a raw ImportError/ModuleNotFoundError
    try:
        nM(cosmo, m_arr, 1.0)
    except ccl.CCLError:
        pass  # expected
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Got bare ImportError/ModuleNotFoundError instead of CCLError")
