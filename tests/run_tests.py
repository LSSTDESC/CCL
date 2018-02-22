from numpy.testing import run_module_suite
import sys

# Per-module accuracy, input correctness, and unit tests
from ccl_test_distances import *
#from ccl_test_growth import *
#from ccl_test_core import *
#from ccl_test_power import *

# Overall interface functionality tests
#from ccl_test_pyccl_interface import *

if __name__ == "__main__":
    # Run all tests
    run_module_suite(argv=sys.argv)
