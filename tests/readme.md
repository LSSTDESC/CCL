# Writing CCL tests

CCL uses the CTest framework (https://github.com/bvdberg/ctest) to write its unit tests. Although lacking a few of the fancy features of more comprehensive frameworks it is lightweight and can be completely included with the CCL, in the file tests/ctest.h

To add a new set of tests to the system:

 - Create a new file tests/your_tests.c
 - Add your tests to that file.  You do not need a "main" function. See the examples in tests/ccl_test_params.c for how to write tests, which should use the ctest assertions to check they are working (see the sparse documentation at https://github.com/bvdberg/ctest).
 - In the Makefile add tests/your_tests.c to the line starting "TESTS="
 - Run "make check" to run all your tests
