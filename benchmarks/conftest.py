"""Config file for pytest. Allow caching for faster test completion.
`$ pytest tests/ --use-cache`
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--use-cache", action="store_true", help="Enable cache.")


def pytest_generate_tests(metafunc):
    if metafunc.config.getoption("use_cache"):
        import pyccl
        pyccl.Caching.enable()


def _isitgr_unusable_reason():
    """Return why isitgr cannot be used (None if it can). An installed
    but unloadable isitgr (e.g. an invalid shared library) raises OSError,
    not ImportError."""
    try:
        import isitgr  # noqa: F401
    except (ImportError, OSError) as exc:
        return f"isitgr is not usable: {exc!r}"
    return None


_ISITGR_REASON = _isitgr_unusable_reason()
_USES_ISITGR = {}


def _module_uses_isitgr(path):
    # The isitgr transfer function is requested inside the tests, not through
    # a pytest parameter, so look at the source of the test module.
    if path not in _USES_ISITGR:
        _USES_ISITGR[path] = "boltzmann_isitgr" in path.read_text(
            encoding="utf-8")
    return _USES_ISITGR[path]


def pytest_collection_modifyitems(config, items):
    # Skip the benchmarks that need a working isitgr when it is unavailable.
    if _ISITGR_REASON is None:
        return
    skip = pytest.mark.skip(reason=_ISITGR_REASON)
    for item in items:
        if _module_uses_isitgr(item.path):
            item.add_marker(skip)
