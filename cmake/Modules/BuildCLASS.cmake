include(ExternalProject)

# Old versions of cmake don't seem to play nice with the GIT_SHALLOW option
if(${CMAKE_VERSION} VERSION_GREATER "3.10.0")
  set(SHALLOW_GIT_CLONE GIT_SHALLOW 1)
endif()

set(CLASSTag v2.6.3)

# In case the compiler being used  is clang, remove the omp flag
if ("${CMAKE_C_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  # using Clang, disabling OpenMP support
  set(CLASS_OMPFLAG "#OMPFLAG   = -fopenmp")
else()
  set(CLASS_OMPFLAG "OMPFLAG   = -fopenmp")
endif()

# Define class install path and sscape the slashes for the Perl command
STRING(REPLACE "/" "\\/" CLASS_INSTALL_DIR "__CLASSDIR__='\"${CMAKE_INSTALL_PREFIX}/share/ccl\"'")

# Downloads and compiles CLASS
ExternalProject_Add(CLASS
        PREFIX CLASS
        GIT_REPOSITORY https://github.com/lesgourg/class_public.git
        GIT_TAG ${CLASSTag}
        ${SHALLOW_GIT_CLONE}
        DOWNLOAD_NO_PROGRESS 1
        PATCH_COMMAND patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/cmake/class-2.6.3.patch
        # In the configuration step, we comment out the default compiler and
        # provide an appropriate omp flag
        CONFIGURE_COMMAND     perl -pi -e "s/^CC /# CC /" Makefile &&
                              perl -pi -e "s/^OMPFLAG .*/${CLASS_OMPFLAG}/" Makefile &&
                              perl -pi -e "s/__CLASSDIR__.*/${CLASS_INSTALL_DIR}/" Makefile
        BUILD_COMMAND         make CC=${CMAKE_C_COMPILER} libclass.a
        INSTALL_COMMAND       mkdir -p ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp libclass.a ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp -r include ${CMAKE_BINARY_DIR}/extern &&
                              mkdir -p ${CMAKE_BINARY_DIR}/extern/share/class/hyrec &&
                              cp hyrec/Alpha_inf.dat ${CMAKE_BINARY_DIR}/extern/share/class/hyrec &&
                              cp hyrec/R_inf.dat ${CMAKE_BINARY_DIR}/extern/share/class/hyrec &&
                              cp hyrec/two_photon_tables.dat ${CMAKE_BINARY_DIR}/extern/share/class/hyrec &&
                              mkdir -p ${CMAKE_BINARY_DIR}/extern/share/class/bbn &&
                              cp bbn/sBBN_2017.dat ${CMAKE_BINARY_DIR}/extern/share/class/bbn
        BUILD_IN_SOURCE 1)
set(CLASS_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
set(CLASS_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include/)
set(CLASS_LIBRARIES -lclass)

# We also need to make sure the CLASS parameter files are added to the install
set(EXTRA_DIST_DIRS ${EXTRA_DIST_DIRS} ${CMAKE_BINARY_DIR}/extern/share/class/)
