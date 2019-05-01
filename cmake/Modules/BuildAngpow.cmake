include(ExternalProject)

# Old versions of cmake don't seem to play nice with the GIT_SHALLOW option
if(${CMAKE_VERSION} VERSION_GREATER "3.10.0")
  set(SHALLOW_GIT_CLONE GIT_SHALLOW 1)
endif()

set(AngpowTag v0.4.2)

# Downloads and compiles Angpow
ExternalProject_Add(ANGPOW
        PREFIX ANGPOW
        GIT_REPOSITORY https://github.com/LSSTDESC/Angpow4CCL.git
        GIT_TAG ${AngpowTag}
        ${SHALLOW_GIT_CLONE}
        DOWNLOAD_NO_PROGRESS 1
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern
                   -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/extern/lib/pkgconfig
                   -DFFTW_USE_STATIC_LIBS=${FFTW_USE_STATIC_LIBS})
set(ANGPOW_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
set(ANGPOW_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include/AngPow)
set(ANGPOW_LIBRARIES -langpow -lstdc++)
