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

    #find_package(PythonInterp)
    find_package(PythonLibsNew)
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
