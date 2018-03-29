cmake_minimum_required(VERSION 3.2)

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(ccl VERSION 0.2.1)

    # Uses system libraries or downloads and build if necessary
    include(BuildFFTW)
    include(BuildGSL)

    # If a CLASS installation folder is provided, use it,
    # otherwise download and compile CLASS
    if(EXTERNAL_CLASS_PATH)
      message("Using external install of CLASS")
      set(CLASS_LIBRARY_DIRS ${EXTERNAL_CLASS_PATH})
      set(CLASS_INCLUDE_DIRS ${EXTERNAL_CLASS_PATH}/include)
      set(CLASS_LIBRARIES -lclass)
      set(CLASS_EXTERNAL True)
    else(EXTERNAL_CLASS_PATH)
      include(BuildCLASS)
    endif(EXTERNAL_CLASS_PATH)

    # Compilation flags
    if ("${CMAKE_C_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
      # using Clang, disabling OpenMP support
      set(CMAKE_C_FLAGS "-O3 -fomit-frame-pointer -fno-common -std=c99 -fPIC -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    else()
      set(CMAKE_C_FLAGS "-O3 -fomit-frame-pointer -fno-common -std=c99 -fPIC -fopenmp -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    endif()

    # Defines list of CCL src files
    set(CCL_SRC src/ccl_background.c src/ccl_core.c src/ccl_error.c src/ccl_lsst_specs.c src/ccl_placeholder.c
                src/ccl_power.c src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c src/ccl_neutrinos.c
                src/ccl_emu17.c src/ccl_correlation.c src/fftlog.c)

    # Defines list of CCL tests src files
    set(TEST_SRC tests/ccl_test.c tests/ccl_test_utils.c tests/ccl_test_params.c tests/ccl_test_cosmology.c
                 tests/ccl_test_distances.c tests/ccl_test_distances_hiz.c tests/ccl_test_growth.c
                 tests/ccl_test_growth_hiz.c tests/ccl_test_bbks.c tests/ccl_test_eh.c tests/ccl_test_cls.c
                 tests/ccl_test_cmblens.c tests/ccl_test_sigmaM.c tests/ccl_test_massfunc.c tests/ccl_test_correlation.c
                 tests/ccl_test_bcm.c tests/ccl_test_emu.c tests/ccl_test_emu_nu.c)

    # Define include and library directories for external dependencies
    include_directories(${CLASS_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS} ${FFTW_INCLUDES})
    link_directories(${CLASS_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} ${FFTW_INCLUDE_DIRS})
    # Adds path to CCL include folder
    include_directories(include)

    #
    # Builds the main CCL library
    #
    # The reason for building ccl as a shared library is that we can link it to
    # class directly, and it's not a dependency anymore
    add_library(ccl SHARED ${CCL_SRC})
    target_link_libraries(ccl ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${CLASS_LIBRARIES} m)
    if(NOT CLASS_EXTERNAL)
      add_dependencies(ccl CLASS)
    endif()
    if(GSL_FOUND)
      set(PCGSL_LIB "-lgsl -lgslcblas")
    else(GSL_FOUND)
      add_dependencies(ccl GSL)
    endif(GSL_FOUND)
    if(FFTW_FOUND)
      set(PCFFTW_LIB "-lfftw3")
    else(FFTW_FOUND)
      add_dependencies(ccl FFTW)
    endif(FFTW_FOUND)

    # Builds the test suite
    add_executable(check_ccl ${TEST_SRC})
    target_link_libraries(check_ccl ccl)

    # Builds pkgconfig file for CCL
    SET(PROJECT_DESCRIPTION "DESC Core Cosmology Library: cosmology routines with validated numerical accuracy")
    SET(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
    SET(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include/")
    SET(PKG_CONFIG_LIBS "-L${PKG_CONFIG_LIBDIR} -lccl ${PCGSL_LIB} ${PCFFTW_LIB} -lm")
    SET(PKG_CONFIG_CFLAGS "-I${PKG_CONFIG_INCLUDEDIR}")
    CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.cmakein"
                    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" )

    # Installs the CCL library
    install(TARGETS ccl DESTINATION lib)
    install(TARGETS check_ccl DESTINATION bin)
    install(DIRECTORY include/ DESTINATION include/
            FILES_MATCHING PATTERN "*.h")
    install(FILES include/ccl_params.ini DESTINATION share/ccl)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ccl.pc DESTINATION lib/pkgconfig)

    #
    # Builds the python module
    #
    include(BuildSWIG)
    include(UseSWIG)

    find_package(PythonLibsNew)
    find_package(NumPy)
    include_directories(BEFORE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

    # Builds swig python module in place
    if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
      swig_add_module(ccllib python pyccl/ccl.i)
    else()
      swig_add_library(ccllib TYPE SHARED LANGUAGE python SOURCES pyccl/ccl.i)
    endif()
    swig_link_libraries(ccllib ${PYTHON_LIBRARIES} ccl)
    set_target_properties(${SWIG_MODULE_ccllib_REAL_NAME} PROPERTIES SUFFIX .so)

    # Forces build of swig executable if not found
    if(NOT SWIG_FOUND)
      add_dependencies(${SWIG_MODULE_ccllib_REAL_NAME} SWIG)
    endif()
