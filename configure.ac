#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.4])
AC_INIT([CCL], [0.0], [github.com/LSSTDESC/CCL])
AM_INIT_AUTOMAKE([-Wall subdir-objects foreign serial-tests])
AM_PROG_AR
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_CONFIG_SRCDIR([src/ccl_core.c])
AC_CONFIG_HEADERS([config.h])

# Avoid problem with git timestamps
AM_MAINTAINER_MODE

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_CC_STDC
AC_PROG_INSTALL
AM_PROG_CC_C_O
AC_PROG_CXX_C_O
AM_MAINTAINER_MODE
AC_OPENMP

# Checks for libraries.
AC_SEARCH_LIBS([pow],[m],[],AC_MSG_ERROR(Couldn't find libm. Required.))
AC_SEARCH_LIBS([cblas_dgemm],[gslcblas],[],AC_MSG_ERROR(Couldn't find libgslcblas.))
AC_SEARCH_LIBS([gsl_spline2d_eval_e],[gsl],[],AC_MSG_ERROR(Couldn't find libgsl. GSL V2+ required.))

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h math.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T
AC_FUNC_MALLOC

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])

AC_CONFIG_FILES([Makefile
		 include/Makefile
		 class/Makefile
		 angpow/Makefile])

AC_OUTPUT

echo \
"
-------------------------------------------------------
 ${PACKAGE_NAME} Version ${PACKAGE_VERSION}

 Configuration settings:
 
 - Prefix: '${prefix}'
 - Compiler: '${CC} ${CFLAGS} ${CPPFLAGS}'

 Now type 'make @<:@<target>@:>@', where the optional
 target may be:

   all     : to build everything (currently libccl)
             this is the default target
   check   : runs the library's unit tests   
   install : to put everything in the installation
             directories ('include', 'lib' and 
	     'bin' from prefix). Root privileges may
	     be necessary for this.
   clean   : to remove garbage
------------------------------------------------------
"
