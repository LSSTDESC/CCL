cmake_minimum_required(VERSION 3.0)
# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(ccl)

    # Finds dependencies
    find_package(FFTW REQUIRED)

    # Builds GSL and CLASS if necessary
    include(BuildGSL)
    include(BuildCLASS)

    # Compilation flags
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
      # using Clang, disabling OpenMP support
      set(CMAKE_C_FLAGS "-O3 -fomit-frame-pointer -fno-common -std=c99 -fPIC -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    else()
      set(CMAKE_C_FLAGS "-O3 -fomit-frame-pointer -fno-common -std=c99 -fPIC -fopenmp -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    endif()

    # Defines list of CCL src files
    set(CCL_SRC src/ccl_background.c
                src/ccl_core.c
                src/ccl_error.c
                src/ccl_lsst_specs.c
                src/ccl_placeholder.c
                src/ccl_power.c
                src/ccl_utils.c
                src/ccl_cls.c
                src/ccl_massfunc.c
                src/ccl_neutrinos.c
                src/ccl_emu17.c
                src/ccl_correlation.c
                src/fftlog.c)

    # Defines list of CCL tests src files
    set(TEST_SRC tests/ccl_test.c
                 tests/ccl_test_utils.c
                 tests/ccl_test_params.c
                 tests/ccl_test_cosmology.c
                 tests/ccl_test_distances.c
                 tests/ccl_test_distances_hiz.c
                 tests/ccl_test_growth.c
                 tests/ccl_test_growth_hiz.c
                 tests/ccl_test_bbks.c
                 tests/ccl_test_eh.c
                 tests/ccl_test_cls.c
                 tests/ccl_test_cmblens.c
                 tests/ccl_test_sigmaM.c
                 tests/ccl_test_massfunc.c
                 tests/ccl_test_correlation.c
                 tests/ccl_test_bcm.c
                 tests/ccl_test_emu.c
                 tests/ccl_test_emu_nu.c)

    # Define include and library directories for external dependencies
    include_directories( ${CLASS_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS})
    link_directories(${CLASS_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} ${FFTW_INCLUDE_DIRS})
    # Adds path to CCL include folder
    include_directories(include)

    # Builds the main CCL library
    add_library(ccl ${CCL_SRC})
    add_dependencies(ccl CLASS)
    if(NOT GSL_FOUND)
      add_dependencies(ccl GSL)
    endif()
    target_link_libraries(ccl ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${CLASS_LIBRARIES} m)

    # Builds the test suite
    add_executable(check_ccl ${TEST_SRC})
    target_link_libraries(check_ccl ccl)

    # Installs the CCL library
    install(TARGETS ccl DESTINATION lib)
    install(TARGETS check_ccl DESTINATION bin)
    install(DIRECTORY include/ DESTINATION include/
            FILES_MATCHING PATTERN "*.h")
    install(FILES include/ccl_params.ini DESTINATION share/ccl)

    # Builds the python module
    include(BuildSWIG)
    include(UseSWIG)

    find_package(PythonInterp)
    set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION_STRING})
    # Introspect the python to get the correct environment
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
"from distutils import sysconfig as s;import sys;import struct;
print('.'.join(str(v) for v in sys.version_info));
print(s.PREFIX);
print(s.get_python_inc(plat_specific=True));
print(s.get_python_lib(plat_specific=True));
print(s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE _PYTHON_VALUES
        ERROR_VARIABLE _PYTHON_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT _PYTHON_SUCCESS MATCHES 0)
        set(PYTHONLIBS_FOUND FALSE)
        return()
    endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
list(GET _PYTHON_VALUES 1 PYTHON_PREFIX)
list(GET _PYTHON_VALUES 2 PYTHON_INCLUDE_DIR)
list(GET _PYTHON_VALUES 3 PYTHON_SITE_PACKAGES)
list(GET _PYTHON_VALUES 4 PYTHON_MODULE_EXTENSION)
list(GET _PYTHON_VALUES 5 PYTHON_IS_DEBUG)
list(GET _PYTHON_VALUES 6 PYTHON_SIZEOF_VOID_P)

    message(${PYTHON_INCLUDE_DIR})
    message(${PYTHON_PREFIX})
    message(${PYTHON_SITE_PACKAGES})

    find_package(PythonLibs)
    find_package(NumPy REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    # Builds swig python module in place
    if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
      swig_add_module(ccllib  python pyccl/ccl.i)
    else()
      swig_add_library(ccllib TYPE SHARED LANGUAGE python SOURCES pyccl/ccl.i)
    endif()
    swig_link_libraries(ccllib ${PYTHON_LIBRARIES} ccl)
    # Forces build of swig executable if not found
    if(NOT SWIG_FOUND)
      add_dependencies(${SWIG_MODULE_ccllib_REAL_NAME} SWIG)
    endif()
