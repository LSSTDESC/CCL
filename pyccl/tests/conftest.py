"""Config file for pytest. Allow caching for faster test completion.
`$ pytest tests/ --use-cache`
"""
from .test_cclobject import all_subclasses, init_decorator
import pyccl


def pytest_addoption(parser):
    parser.addoption("--use-cache", action="store_true", help="Enable cache.")


def pytest_generate_tests(metafunc):
    if metafunc.config.getoption("use_cache"):
        import pyccl
        pyccl.Caching.enable()


# For testing, we want to make sure that all instances of the subclasses
# of `CCLAutoRepr` contain all attributes listed in `__repr_attrs__`.
# We run some things post-init for these subclasses, which are triggered
# during smoke tests.
for sub in list(all_subclasses(pyccl.CCLAutoRepr)):
    sub.__init__ = init_decorator(sub.__init__)
