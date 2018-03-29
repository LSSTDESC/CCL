include(ExternalProject)

set(CLASSVersion 2.6.3)
set(CLASSMD5 e6eb0fd721bb1098e642f5d1970501ce)

# In case the compiler being used  is clang, remove the omp flag
if ("${CMAKE_C_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  # using Clang, disabling OpenMP support
  set(CLASS_OMPFLAG "#OMPFLAG   = -fopenmp")
else()
  set(CLASS_OMPFLAG "OMPFLAG   = -fopenmp")
endif()


# Downloads and compiles CLASS
ExternalProject_Add(CLASS
        PREFIX CLASS
        URL https://github.com/lesgourg/class_public/archive/v${CLASSVersion}.tar.gz
        URL_MD5 ${CLASSMD5}
        DOWNLOAD_NO_PROGRESS 1
        # In the configuration step, we comment out the default compiler and
        # provide an appropriate omp flag
        CONFIGURE_COMMAND     perl -pi -e "s/^CC /# CC /" Makefile &&
                              perl -pi -e "s/^OMPFLAG .*/${CLASS_OMPFLAG}/" Makefile
        BUILD_COMMAND         make CC=${CMAKE_C_COMPILER} libclass.a
        INSTALL_COMMAND       mkdir -p ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp libclass.a ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp -r include ${CMAKE_BINARY_DIR}/extern
        BUILD_IN_SOURCE 1)
set(CLASS_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
set(CLASS_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include/)
set(CLASS_LIBRARIES -lclass)
