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

