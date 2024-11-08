cmake_minimum_required(VERSION 3.5)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(ccl VERSION 3.1.2)

# Uncomment to get detailed build outputs
# set( CMAKE_VERBOSE_MAKEFILE on )

option(FORCE_OPENMP "Forcibly use OpenMP " NO)

# Defines list of CCL src files
set(CCL_SRC
    src/ccl_background.c src/ccl_core.c src/ccl_error.c
    src/ccl_bbks.c src/ccl_f1d.c src/ccl_f2d.c src/ccl_f3d.c
    src/ccl_power.c
    src/ccl_eh.c src/ccl_musigma.c
    src/ccl_utils.c src/ccl_cls.c src/ccl_massfunc.c
    src/ccl_neutrinos.c
    src/ccl_correlation.c
    src/ccl_halofit.c
    src/ccl_tracers.c
    src/ccl_mass_conversion.c
    src/ccl_fftlog.c)

# Defines list of CCL C test src files
# ! Add new tests of the C code to this list
set(TEST_SRC
    # test core code first
    benchmarks/ccl_test.c
    benchmarks/ccl_test_utils.c
    benchmarks/ccl_test_f2d.c
    benchmarks/ccl_test_f3d.c
)


# Defines list of extra distribution files and directories to be installed on the system
set(EXTRA_DIST README.md LICENSE LICENSE_COSMICEMU)
set(EXTRA_DIST_DIRS)

# Uses system libraries or downloads and build if necessary
include(BuildFFTW)
include(BuildGSL)

if(("${CMAKE_C_COMPILER_ID}" MATCHES "^(Apple)+Clang$") AND (CMAKE_C_COMPILER_VERSION LESS 13) AND NOT FORCE_OPENMP)
    # Apple Clang 12 (and maybe below) have issues with OpenMP and cmake, so we don't try to find it
    # Anything else supports OpenMP
    # This can be overwritten by setting FORCE_OPENMP=TRUE and adding "-L/usr/local/lib -lomp" to LDFLAGS
    # by issuing 'export LDFLAGS="$LDFLAGS -L/usr/local/lib -lomp"' in the command line
else()
    find_package(OpenMP)
endif()

# Compilation flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -fomit-frame-pointer -fno-common -fPIC -std=gnu99")
set(CMAKE_C_FLAGS_DEBUG   "-O0 -g -fomit-frame-pointer -fno-common -fPIC -std=gnu99")

if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    include_directories(${OpenMP_C_INCLUDE_DIR})
    if(FORCE_OPENMP)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${OpenMP_C_FLAGS}")
    endif()
endif()

if(DEFINED ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/include")
    add_link_options("LINKER:-rpath,$ENV{CONDA_PREFIX}/lib")
    # Use conda libraries if available
    add_link_options("-L$ENV{CONDA_PREFIX}/lib")
endif()

# Define include and library directories for external dependencies
include_directories(${GSL_INCLUDE_DIRS} ${FFTW_INCLUDES})
link_directories(${GSL_LIBRARY_DIRS} ${FFTW_LIBRARY_DIRS})
# Adds path to CCL include folder
include_directories(include)

#
# Builds the main CCL library
#

# Compiles all the source files
add_library(objlib OBJECT ${CCL_SRC})
# Make sure the external projects are correclty built
if(NOT GSL_FOUND)
    add_dependencies(objlib GSL)
endif()
if(NOT FFTW_FOUND)
    add_dependencies(objlib FFTW)
endif()

# The reason for building ccl as a shared library is that we can link it to
# class directly, and it's not a dependency anymore
add_library(ccl SHARED $<TARGET_OBJECTS:objlib>)
# The generator expression $<$<BOOL:${OpenMP_C_FOUND}>:OpenMP::OpenMP_C> adds the OpenMP library if OpenMP is available
target_link_libraries(ccl ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${CLASS_LIBRARIES} m $<$<BOOL:${OpenMP_C_FOUND}>:OpenMP::OpenMP_C>)

# Builds the test suite
add_executable(check_ccl ${TEST_SRC})
target_link_libraries(check_ccl ccl)

# Builds pkgconfig file for CCL
SET(PROJECT_DESCRIPTION "DESC Core Cosmology Library: cosmology routines with validated numerical accuracy")
SET(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
SET(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include/")
SET(PKG_CONFIG_REQUIRES "gsl >= 2.1")
SET(PKG_CONFIG_REQUIRES_PRIVATE "fftw3")
SET(PKG_CONFIG_LIBS "-L${PKG_CONFIG_LIBDIR} -lccl -lm")
SET(PKG_CONFIG_CFLAGS "-I${PKG_CONFIG_INCLUDEDIR}")
CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.cmakein"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" )

# Installs the CCL library
install(
    TARGETS ccl check_ccl
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(
    DIRECTORY include/
    DESTINATION include
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
