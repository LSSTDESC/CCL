"""
python module for calculating integrals with 1 Bessel functions.
This module contains:
-- FFTLog method for integrals with 1 spherical Bessel function;
-- integrals with 1 (cylindrical) Bessel function, i.e. Hankel transform;
-- window function (optional) for smoothing Fourier coefficients

by Xiao Fang
Apr 10, 2019
"""

import numpy as np
from lib_wrapper import *

class fftlog(object):

	def __init__(self, x, fx, nu=1.1, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):

		self.x_origin = x # x is logarithmically spaced
		# self.lnx = np.log(x)
		self.dlnx = np.log(x[1]/x[0])
		self.fx_origin= fx # f(x) array
		self.nu = nu
		self.N_extrap_low = N_extrap_low
		self.N_extrap_high = N_extrap_high
		self.c_window_width = c_window_width

		# extrapolate x and f(x) linearly in log(x), and log(f(x))
		self.x = log_extrap(x, N_extrap_low, N_extrap_high)
		self.fx = log_extrap(fx, N_extrap_low, N_extrap_high)
		self.N = self.x.size

		# zero-padding
		self.N_pad = N_pad
		if(N_pad):
			pad = np.zeros(N_pad)
			self.x = log_extrap(self.x, N_pad, N_pad)
			self.fx = np.hstack((pad, self.fx, pad))
			self.N += 2*N_pad
			self.N_extrap_high += N_pad
			self.N_extrap_low += N_pad

		if(self.N%2==1): # Make sure the array sizes are even
			self.x= self.x[:-1]
			self.fx=self.fx[:-1]
			self.N -= 1
			if(N_pad):
				self.N_extrap_high -=1

	# 	self.m, self.c_m = self.get_c_m()
	# 	self.eta_m = 2*np.pi/(float(self.N)*self.dlnx) * self.m


	# def get_c_m(self):
	# 	"""
	# 	return m and c_m
	# 	c_m: the smoothed FFT coefficients of "biased" input function f(x): f_b = f(x) / x^\nu
	# 	number of x values should be even
	# 	c_window_width: the fraction of c_m elements that are smoothed,
	# 	e.g. c_window_width=0.25 means smoothing the last 1/4 of c_m elements using "c_window".
	# 	"""

	# 	f_b=self.fx * self.x**(-self.nu)
	# 	c_m=rfft(f_b)
	# 	m=np.arange(0,self.N//2+1) 
	# 	c_m = c_m*c_window(m, int(self.c_window_width*self.N//2.) )
	# 	return m, c_m

	def _fftlog(self, ell, derivative):
		"""
		cfftlog wrapper
		"""
		y = np.zeros(self.N)
		Fy= np.zeros(self.N)
		# print(self.x, self.x.shape)
		cfftlog_wrapper(np.ascontiguousarray(self.x, dtype=np.float64),
						np.ascontiguousarray(self.fx, dtype=np.float64),
						clong(self.N),
						cdouble(ell),
						np.ascontiguousarray(y, dtype=np.float64),
						np.ascontiguousarray(Fy, dtype=np.float64),
						cdouble(self.nu),
						cdouble(self.c_window_width),
						cint(derivative),
						clong(0) # Here N_pad is done in python, not in C
						)

		return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

	def _fftlog_ells(self, ell_array, derivative):
		"""
		cfftlog_ells wrapper
		"""
		Nell = ell_array.shape[0]
		y = np.zeros((Nell,self.N))
		Fy= np.zeros((Nell,self.N))
		ypp = (y.__array_interface__['data'][0] 
      			+ np.arange(Nell)*y.strides[0]).astype(np.uintp) 
		Fypp = (Fy.__array_interface__['data'][0] 
				+ np.arange(Nell)*Fy.strides[0]).astype(np.uintp)

		cfftlog_ells_wrapper(np.ascontiguousarray(self.x, dtype=np.float64),
						np.ascontiguousarray(self.fx, dtype=np.float64),
						clong(self.N),
						np.ascontiguousarray(ell_array, dtype=np.float64),
						clong(Nell),
						ypp,
						Fypp,
						cdouble(self.nu),
						cdouble(self.c_window_width),
						cint(derivative),
						clong(0) # Here N_pad is done in python, not in C
						)

		return y[:,self.N_extrap_high:self.N-self.N_extrap_low], Fy[:,self.N_extrap_high:self.N-self.N_extrap_low]

	def _fftlog_modified_ells(self, ell_array, derivative):
		"""
		cfftlog_ells wrapper
		"""
		Nell = ell_array.shape[0]
		y = np.zeros((Nell,self.N))
		Fy= np.zeros((Nell,self.N))
		ypp = (y.__array_interface__['data'][0] 
      			+ np.arange(Nell)*y.strides[0]).astype(np.uintp) 
		Fypp = (Fy.__array_interface__['data'][0] 
				+ np.arange(Nell)*Fy.strides[0]).astype(np.uintp)

		cfftlog_modified_ells_wrapper(np.ascontiguousarray(self.x, dtype=np.float64),
						np.ascontiguousarray(self.fx, dtype=np.float64),
						clong(self.N),
						np.ascontiguousarray(ell_array, dtype=np.float64),
						clong(Nell),
						ypp,
						Fypp,
						cdouble(self.nu),
						cdouble(self.c_window_width),
						cint(derivative),
						clong(0) # Here N_pad is done in python, not in C
						)

		return y[:,self.N_extrap_high:self.N-self.N_extrap_low], Fy[:,self.N_extrap_high:self.N-self.N_extrap_low]


	def fftlog(self, ell):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog(ell, 0)

	def fftlog_dj(self, ell):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j'_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog(ell, 1)

	def fftlog_ddj(self, ell):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j''_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog(ell, 2)

	def fftlog_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_ells(ell_array, 0)

	def fftlog_dj_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j'_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_ells(ell_array, 1)

	def fftlog_ddj_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x) * j''_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_ells(ell_array, 2)

	def fftlog_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 0)

	def fftlog_dj_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j'_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 1)

	def fftlog_ddj_modified_ells(self, ell_array):
		"""
		Calculate F(y) = \int_0^\infty dx / x * f(x)/(xy)^2 * j''_\ell(xy),
		where j_\ell is the spherical Bessel func of order ell.
		array y is set as y[:] = (ell+1)/x[::-1]
		"""
		return self._fftlog_modified_ells(ell_array, 2)


class hankel(object):
	def __init__(self, x, fx, nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
		print('nu is required to be between (0.5-n) and 2.')
		self.myfftlog = fftlog(x, np.sqrt(x)*fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)
	
	def hankel(self, n):
		y, Fy = self.myfftlog.fftlog(n-0.5)
		Fy *= np.sqrt(2*y/np.pi)
		return y, Fy

	def hankel_narray(self, n_array):
		y, Fy = self.myfftlog.fftlog_ells(n_array-0.5)
		Fy *= np.sqrt(2*y/np.pi)
		return y, Fy


### Utility functions ####################

def log_extrap(x, N_extrap_low, N_extrap_high):

	low_x = high_x = []
	if(N_extrap_low):
		dlnx_low = np.log(x[1]/x[0])
		low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0) )
	if(N_extrap_high):
		dlnx_high= np.log(x[-1]/x[-2])
		high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high+1) )
	x_extrap = np.hstack((low_x, x, high_x))
	return x_extrap
