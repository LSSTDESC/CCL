from numpy.testing import run_module_suite
import sys

# Per-module accuracy, input correctness, and unit tests
from ccl_test_distances import *
from ccl_test_growth import *
from ccl_test_core import *
from ccl_test_power import *
from ccl_test_cclerror import *
from ccl_test_pk2d import  *

# Overall interface functionality tests
from ccl_test_pyccl_interface import *
from ccl_test_swig_interface import *

if __name__ == "__main__":
    # Run tests
    args = sys.argv

    # If no args were specified, add arg to only do non-slow tests
    if len(args) == 1:
        print("Running tests that are not tagged as 'slow'. "
              "Use '--all' to run all tests.")
        args.append("-a!slow")
    run_module_suite(argv=args)
