"""Config file for pytest. Allow caching for faster test completion.
`$ pytest tests/ --use-cache`
"""

def pytest_addoption(parser):
    parser.addoption("--use-cache", action="store_true", help="Enable cache.")

    parser.addoption(
        "--plot-ccl-bench",
        action="store_true",
        default=False,
        help="Save CCL vs benchmark correlation plots.",
    )
    
    parser.addoption(
        "--plot-dir",
        action="store",
        default="benchmarks/plots",
        help="Directory where plots are saved (used with --plot-ccl-bench).",
    )



def pytest_generate_tests(metafunc):
    if metafunc.config.getoption("use_cache"):
        import pyccl
        pyccl.Caching.enable()
