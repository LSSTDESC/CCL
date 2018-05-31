#
# Builds the python module
#

include(BuildSWIG)
include(UseSWIG)

find_package(PythonLibsNew)
find_package(NumPy)
include_directories(BEFORE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# We also build a static library only for linking the python module
add_library(ccl_static STATIC $<TARGET_OBJECTS:objlib>)
target_link_libraries(ccl_static ${GSL_LIBRARIES} ${CLASS_LIBRARIES} ${ANGPOW_LIBRARIES} ${FFTW_LIBRARIES} m)
add_dependencies(ccl_static ccl)

# Builds swig python module in place
if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  swig_add_module(ccllib python ccl.i)
else()
  swig_add_library(ccllib TYPE SHARED LANGUAGE python SOURCES ccl.i)
endif()
swig_link_libraries(ccllib ccl_static)
if(APPLE)
    # Unpleasant subtelty for linking on osx
    set_target_properties(${SWIG_MODULE_ccllib_REAL_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else(APPLE)
    swig_link_libraries(ccllib ${PYTHON_LIBRARIES})
endif()
set_target_properties(${SWIG_MODULE_ccllib_REAL_NAME} PROPERTIES SUFFIX .so)

# Forces build of swig executable if not found
if(NOT SWIG_FOUND)
  add_dependencies(${SWIG_MODULE_ccllib_REAL_NAME} SWIG)
endif()
