include(ExternalProject)

set(FFTWVersion 3.3.7)
set(FFTWMD5 0d5915d7d39b3253c1cc05030d79ac47)

find_package(FFTW)

# If FFTW is not installed, lets go ahead and compile it
if(NOT FFTW_FOUND )
    message(STATUS "FFTW not found, downloading and compiling from source")
    ExternalProject_Add(FFTW
        PREFIX FFTW
        URL http://www.fftw.org/fftw-${FFTWVersion}.tar.gz
        URL_MD5 ${GSLMD5}
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/extern --enable-shared=no --with-pic=yes
        BUILD_COMMAND           make -j8
        INSTALL_COMMAND         make install
        BUILD_IN_SOURCE 1)
        set(FFTW_USE_STATIC_LIBS TRUE)
        set(FFTW_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
        set(FFTW_INCLUDES ${CMAKE_BINARY_DIR}/extern/include/)
        set(FFTW_LIBRARIES -lfftw3)
endif()
