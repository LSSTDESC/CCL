"""Config file for pytest. Allow caching for faster test completion.
`$ pytest tests/ --use-cache`
"""
import pytest

from .test_cclobject import all_subclasses, init_decorator
import pyccl


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


def pytest_collection_modifyitems(config, items):
    # Skip the tests that need a working isitgr when it is unavailable.
    if _ISITGR_REASON is None:
        return
    skip = pytest.mark.skip(reason=_ISITGR_REASON)
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        # Only compare string parameters: other parameters may be arrays,
        # for which `==` would not return a single bool.
        if any(isinstance(val, str) and val == "boltzmann_isitgr"
               for val in callspec.params.values()):
            item.add_marker(skip)


# For testing, we want to make sure that all instances of the subclasses
# of `CCLAutoRepr` contain all attributes listed in `__repr_attrs__`.
# We run some things post-init for these subclasses, which are triggered
# during smoke tests.
for sub in list(all_subclasses(pyccl.CCLAutoRepr)):
    sub.__init__ = init_decorator(sub.__init__)
