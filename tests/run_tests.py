from numpy.testing import run_module_suite

# Per-module accuracy and input correctness tests
from ccl_test_distances import *
from ccl_test_growth import *
from ccl_test_core import *

# Overall interface functionality tests
from ccl_test_pyccl_interface import *

if __name__ == "__main__":
    # Run all tests
    run_module_suite()
