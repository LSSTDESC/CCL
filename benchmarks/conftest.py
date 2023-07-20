"""Config file for pytest. Allow caching for faster test completion.
`$ pytest tests/ --use-cache`
"""


def pytest_addoption(parser):
    parser.addoption("--use-cache", action="store_true", help="Enable cache.")


def pytest_generate_tests(metafunc):
    if metafunc.config.getoption("use_cache"):
        import pyccl
        pyccl.Caching.enable()
