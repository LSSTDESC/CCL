cmake_minimum_required(VERSION 3.2)
if (NOT CMAKE_BUILD_TYPE)
		message(STATUS "No build type selected, default to Release")
		set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(ccl VERSION 0.2.1)

    # Defines list of CCL src files
    set(CCL_SRC src/ccl_background.c src/ccl_params.c src/ccl_core.c src/ccl_error.c src/ccl_lsst_specs.c
                src/ccl_p2d.c src/ccl_power.c src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c src/ccl_neutrinos.c
              src/ccl_emu17.c src/ccl_correlation.c src/ccl_halomod.c src/fftlog.c)

    # Defines list of CCL tests src files
    # ! Add new tests to this list
    set(TEST_SRC tests/ccl_test.c tests/ccl_test_utils.c tests/ccl_test_params.c tests/ccl_test_params_mnu.c
	             tests/ccl_test_cosmology.c
    		 tests/ccl_test_distances_class_allz.c tests/ccl_test_distances_cosmomad_hiz.c tests/ccl_test_distances_cosmomad_lowz.c
                 tests/ccl_test_distances_astropy_mnu_lowz.c tests/ccl_test_distances_astropy_mnu_hiz.c
		 tests/ccl_test_growth_lowz.c tests/ccl_test_growth_hiz.c tests/ccl_test_growth_allz.c
                 tests/ccl_test_bbks.c tests/ccl_test_eh.c
		 tests/ccl_test_cls.c tests/ccl_test_sigmaM.c
		 tests/ccl_test_massfunc.c tests/ccl_test_correlation.c tests/ccl_test_correlation_3d.c
		 tests/ccl_test_bcm.c tests/ccl_test_emu.c tests/ccl_test_emu_nu.c
		 tests/ccl_test_power_nu.c tests/ccl_test_halomod.c tests/ccl_test_angpow.c
		 tests/ccl_test_p2d.c)


    # Defines list of extra distribution files and directories to be installed on the system
    set(EXTRA_DIST include/ccl_params.ini README.md LICENSE LICENSE_COSMICEMU)
    set(EXTRA_DIST_DIRS)

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

    # Builds Angpow
    # TODO: the same mechanism as for CLASS could be used here to provide the
    # option to use a pre-installed version of angpow
    include(BuildAngpow)

    # Compilation flags
    set(CMAKE_C_FLAGS_RELEASE "-O3 -fomit-frame-pointer -fno-common -fPIC -std=gnu99 -DHAVE_ANGPOW -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    set(CMAKE_C_FLAGS_DEBUG   "-Og -fomit-frame-pointer -fno-common -fPIC -std=gnu99 -DHAVE_ANGPOW -D__CCL_DATA_DIR__=${CMAKE_INSTALL_PREFIX}/share/ccl")
    if (NOT "${CMAKE_C_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
      # When not using Clang and in Release mode, enabling OpenMP support
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fopenmp")
      set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fopenmp")
    endif()

    # Define include and library directories for external dependencies
    include_directories(${CLASS_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS} ${FFTW_INCLUDES} ${ANGPOW_INCLUDE_DIRS})
    link_directories(${CLASS_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} ${FFTW_LIBRARY_DIRS} ${ANGPOW_LIBRARY_DIRS})
    # Adds path to CCL include folder
    include_directories(include)

    #
    # Builds the main CCL library
    #

    # Compiles all the source files
    add_library(objlib OBJECT ${CCL_SRC})
    # Make sure the external projects are correclty built
    add_dependencies(objlib ANGPOW)
    if(NOT CLASS_EXTERNAL)
      add_dependencies(objlib CLASS)
    endif()
    if(NOT GSL_FOUND)
      add_dependencies(objlib GSL)
      add_dependencies(ANGPOW GSL)
    endif()
    if(NOT FFTW_FOUND)
      add_dependencies(objlib FFTW)
      add_dependencies(ANGPOW FFTW)
    endif()

    # The reason for building ccl as a shared library is that we can link it to
    # class directly, and it's not a dependency anymore
    add_library(ccl SHARED $<TARGET_OBJECTS:objlib>)
    target_link_libraries(ccl ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${CLASS_LIBRARIES} ${ANGPOW_LIBRARIES} m)

    # Builds the test suite
    add_executable(check_ccl ${TEST_SRC})
    target_link_libraries(check_ccl ccl)

    # Builds example folder in place, make this optional
    add_subdirectory(examples "${CMAKE_CURRENT_SOURCE_DIR}/examples" EXCLUDE_FROM_ALL)

    # Builds pkgconfig file for CCL
    SET(PROJECT_DESCRIPTION "DESC Core Cosmology Library: cosmology routines with validated numerical accuracy")
    SET(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
    SET(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include/")
    SET(PKG_CONFIG_REQUIRES "gsl >= 2.1")
    SET(PKG_CONFIG_REQUIRES_PRIVATE "fftw3")
    SET(PKG_CONFIG_LIBS "-L${PKG_CONFIG_LIBDIR} -lccl -lm")
    SET(PKG_CONFIG_CFLAGS "-I${PKG_CONFIG_INCLUDEDIR}")
    CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.cmakein"
                    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" )

    # Installs the CCL library
    install(TARGETS ccl check_ccl
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
    install(DIRECTORY include/ DESTINATION include
            FILES_MATCHING PATTERN "*.h")
    install(FILES ${EXTRA_DIST} DESTINATION share/ccl)
    install(DIRECTORY ${EXTRA_DIST_DIRS} DESTINATION share/ccl)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ccl.pc DESTINATION lib/pkgconfig)

    # Adds uninstall target
    if(NOT TARGET uninstall)
      configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
          "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
          IMMEDIATE @ONLY)
      add_custom_target(uninstall
          COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif()

    # Optionally builds the python wrapper
    add_subdirectory(pyccl EXCLUDE_FROM_ALL)
