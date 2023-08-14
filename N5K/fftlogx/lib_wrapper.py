"""


by Xiao Fang
Apr 10, 2019
"""
import sys
import os
import ctypes
import numpy as np


def load_library(libname, path=None):
	if path is None:
		dirname = os.path.split(__file__)[0]
	lib_name = os.path.join(dirname, libname)
	lib=ctypes.cdll.LoadLibrary(lib_name)
	return lib

fftlogx_lib = load_library("../build/libfftlogx.so")

cdouble = ctypes.c_double
cint 	= ctypes.c_int
clong 	= ctypes.c_long

# Based on Stackoverflow: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
_doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

def _array_ctype(ndim, dtype=np.float64, flags="C_CONTIGUOUS"):
    return [np.ctypeslib.ndpointer(ndim=ndim, dtype=dtype, flags=flags)]

cfftlog_wrapper = fftlogx_lib.cfftlog_wrapper
cfftlog_wrapper.argtypes = [*_array_ctype(ndim=1, dtype=np.float64), # x
							*_array_ctype(ndim=1, dtype=np.float64), # fx
							clong, 									 # N
							cdouble,								 # ell
							*_array_ctype(ndim=1, dtype=np.float64), # y
							*_array_ctype(ndim=1, dtype=np.float64), # Fy
							cdouble, 								 # nu
							cdouble, 								 # c_window_width
							cint,									 # derivative (of Bessel)
							clong									 # N_pad
							]



cfftlog_ells_wrapper = fftlogx_lib.cfftlog_ells_wrapper
cfftlog_ells_wrapper.argtypes = [*_array_ctype(ndim=1, dtype=np.float64), # x
							  	 *_array_ctype(ndim=1, dtype=np.float64), # fx
								 clong, 		 						  # N
								 *_array_ctype(ndim=1, dtype=np.float64), # ell array
								 clong,							 	  	  # Nell
								 _doublepp, # y
								 _doublepp, # Fy
								 cdouble, 								  # nu
								 cdouble, 								  # c_window_width
								 cint,								 	  # derivative (of Bessel)
								 clong									  # N_pad
								 ]

cfftlog_modified_ells_wrapper = fftlogx_lib.cfftlog_modified_ells_wrapper
cfftlog_modified_ells_wrapper.argtypes = [*_array_ctype(ndim=1, dtype=np.float64), # x
							  	 *_array_ctype(ndim=1, dtype=np.float64), # fx
								 clong, 		 						  # N
								 *_array_ctype(ndim=1, dtype=np.float64), # ell array
								 clong,							 	  	  # Nell
								 _doublepp, # y
								 _doublepp, # Fy
								 cdouble, 								  # nu
								 cdouble, 								  # c_window_width
								 cint,								 	  # derivative (of Bessel)
								 clong									  # N_pad
								 ]




