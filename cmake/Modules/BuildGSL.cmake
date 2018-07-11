include(ExternalProject)

set(GSLVersion 2.4)
set(GSLMD5 dba736f15404807834dc1c7b93e83b92)

find_package(GSL 2.1)

# If GSL is not installed, lets go ahead and compile it
if(NOT GSL_FOUND )
    message(STATUS "GSL not found, downloading and compiling from source")
    ExternalProject_Add(GSL
        PREFIX GSL
        URL ftp://ftp.gnu.org/gnu/gsl/gsl-${GSLVersion}.tar.gz
        URL_MD5 ${GSLMD5}
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ./configure
                                                            --prefix=${CMAKE_BINARY_DIR}/extern
                                                            --enable-shared=no
                                                            --with-pic=yes
        BUILD_COMMAND           make -j8 > out.log 2>&1
        INSTALL_COMMAND         make install
        BUILD_IN_SOURCE 1)
        set(GSL_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
        set(GSL_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include/)
        set(GSL_LIBRARIES -lgsl -lgslcblas -lm)
endif()
