include(ExternalProject)

set(SWIGVersion 3.0.12)
set(SWIGMD5 1732ea3429cc8b85ab18dd5115dbcfb7)

find_package(SWIG)

# If SWIG is not installed, lets go ahead and compile it
if(NOT SWIG_FOUND )
      message(STATUS "SWIG not found, downloading and compiling from source")
      ExternalProject_Add(SWIG
              PREFIX SWIG
              URL https://github.com/swig/swig/archive/rel-${SWIGVersion}.tar.gz
              URL_MD5 ${SWIGMD5}
              DOWNLOAD_NO_PROGRESS 1
              CONFIGURE_COMMAND     ./autogen.sh && ./configure --prefix=${CMAKE_BINARY_DIR}/extern
              BUILD_COMMAND         make -j8
              INSTALL_COMMAND       make install
              BUILD_IN_SOURCE 1)
      set(SWIG_DIR ${CMAKE_BINARY_DIR}/extern)
      set(SWIG_EXECUTABLE ${CMAKE_BINARY_DIR}/extern/bin/swig)
endif()
