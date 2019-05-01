/* @file
 *
 * Defines useful macros for compilation.
 * */


/* define marks for
 * begin and end of declaration sections of header files for C++ compatibility */

#ifdef __cplusplus

#define CCL_BEGIN_DECLS extern "C" {
#define CCL_END_DECLS }

#else

#define CCL_BEGIN_DECLS
#define CCL_END_DECLS

#endif

/**
 *  PI (in case it's not defined from math.h)
*/
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
