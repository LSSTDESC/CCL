include(ExternalProject)

set(CLASSVersion 2.6.3)
set(CLASSMD5 e6eb0fd721bb1098e642f5d1970501ce)

# Downloads and compiles CLASS
ExternalProject_Add(CLASS
        PREFIX CLASS
        URL https://github.com/lesgourg/class_public/archive/v${CLASSVersion}.tar.gz
        URL_MD5 ${CLASSMD5}
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND     echo "No configuration step for CLASS"
        BUILD_COMMAND         make libclass.a
        INSTALL_COMMAND       mkdir -p ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp libclass.a ${CMAKE_BINARY_DIR}/extern/lib &&
                              cp -ar include ${CMAKE_BINARY_DIR}/extern
        BUILD_IN_SOURCE 1)
set(CLASS_LIBRARY_DIRS ${CMAKE_BINARY_DIR}/extern/lib/ )
set(CLASS_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/extern/include/)
set(CLASS_LIBRARIES -lclass)
