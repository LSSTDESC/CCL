import numpy as np
import pyccl as ccl
from .calculator_base import N5KCalculatorBase
from scipy.interpolate import interp1d
#from fftlogx.fftlogx import *
#from pyccl.pyutils import _fftlog_transform
import time
import sys
from pyccl.pyutils import _fftlog_transform_general
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

class N5KCalculatorFKEM(N5KCalculatorBase):
		name = 'FKEM' # Fang, Krause, Eifler, MacCrann
		def log_extrap(self,x, N_extrap_low, N_extrap_high):

			low_x = high_x = []
			if(N_extrap_low):
				dlnx_low = np.log(x[1]/x[0])
				low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0) )
			if(N_extrap_high):
				dlnx_high= np.log(x[-1]/x[-2])
				high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high+1) )
			x_extrap = np.hstack((low_x, x, high_x))
			return x_extrap

		def pad_array(self,k, pk, N_pad):
			if(N_pad):
				pad = np.zeros(N_pad)
				k = self.log_extrap(k, N_pad, N_pad)
				pk = np.hstack((pad, pk, pad))
			if(len(pk)%2==1): # Make sure the array sizes are even
				k= k[:-1]
				pk=pk[:-1]
			return k, pk

		def unpad_array(self,k, pk, N_pad):
			if(N_pad):
				k = k[(N_pad):]
				k = k[:(-N_pad)]
				pk = pk[(N_pad):]
				pk = pk[:(-N_pad)]
			return k, pk			





		def setup(self):
				# Initialize cosmology
				par = self.get_cosmological_parameters()
				dpk = self.get_pk()
				a = 1./(1+dpk['z'][::-1])
				self.cosmo = ccl.CosmologyCalculator(Omega_c=par['Omega_m']-par['Omega_b'],
																						 Omega_b=par['Omega_b'],
																						 h=par['h'], n_s=par['n_s'],
																						 A_s=par['A_s'], w0=par['w0'],
																						 pk_linear={'a': a,
																												'k': dpk['k'],
																												'delta_matter:delta_matter': dpk['pk_lin'][::-1][:]},
																						 pk_nonlin={'a': a,
																												'k': dpk['k'],
																												'delta_matter:delta_matter': dpk['pk_nl'][::-1][:]})

				self.cosmo_nonlin_part = ccl.CosmologyCalculator(Omega_c=par['Omega_m']-par['Omega_b'],
																						 Omega_b=par['Omega_b'],
																						 h=par['h'], n_s=par['n_s'],
																						 A_s=par['A_s'], w0=par['w0'],
																						 pk_linear={'a': a,
																												'k': dpk['k'],
																												'delta_matter:delta_matter': dpk['pk_lin'][::-1][:]},
																						 # pk_nonlin={'a': a,
																							# 					'k': dpk['k'],
																							# 					'delta_matter:delta_matter': (dpk['pk_lin'][::-1][:])})

																						 pk_nonlin={'a': a,
																												'k': dpk['k'],
																												'delta_matter:delta_matter': (dpk['pk_nl'][::-1][:]-dpk['pk_lin'][::-1][:])})


				chi_min = ccl.comoving_radial_distance(cosmo=self.cosmo, a=1./(1.+0.002))
				chi_max = ccl.comoving_radial_distance(cosmo=self.cosmo, a=1./(1.+4.))
				Nchi = int(self.config['Nchi_fft'])
				self.chi_logspace_arr = np.logspace(np.log10(chi_min),np.log10(chi_max),num=Nchi, endpoint=True)
				self.dlnr = np.log(chi_max/chi_min)/(Nchi-1.)
				a_arr = ccl.scale_factor_of_chi(self.cosmo, self.chi_logspace_arr)
				growfac_arr = ccl.growth_factor(self.cosmo, a_arr)

				self.plin0 = dpk['pk_lin'][0]
				self.k_in = dpk['k']
				self.pklin_interp = interp1d(np.log(self.k_in), np.log(self.plin0), fill_value='extrapolate')

				# Initialize tracers
				if self.config.get('tracers_from_kernels', False):
						tpar = self.get_tracer_parameters()
						ker = self.get_tracer_kernels()
						a_g = 1./(1+ker['z_cl'][::-1])
						self.t_g = []
						self.t_g_logspace = []


						for k in ker['kernels_cl']:
								t = ccl.Tracer()
								barr = np.ones_like(a_g)

								t.add_tracer(self.cosmo,(ker['chi_cl'], k), transfer_a=(a_g, barr))
								self.t_g.append(t)

								ker_at_chi = interp1d(np.log(ker['chi_cl']), k, bounds_error=False, fill_value=0.)
								ker_logspace = ker_at_chi(np.log(self.chi_logspace_arr)) * self.chi_logspace_arr * growfac_arr
								# import matplotlib.pyplot as plt 
								#plt.plot(self.chi_logspace_arr, ker_logspace, label='%d'%(len(self.t_g_logspace)))
								#plt.legend()
								#plt.show()
								self.t_g_logspace.append(ker_logspace)

						self.t_s = []
						self.t_s_logspace = []
						self.ker_chi_interpfuncs = []
						for k in ker['kernels_sh']:
							t = ccl.Tracer()
							t.add_tracer(self.cosmo,kernel=(ker['chi_sh'], k),der_bessel=-1, der_angles=2)
							self.t_s.append(t)

							ker_at_chi = interp1d(np.log(ker['chi_sh']), k,bounds_error=False, fill_value=0.)
							ker_logspace = ker_at_chi(np.log(self.chi_logspace_arr)) * self.chi_logspace_arr * growfac_arr
							# import matplotlib.pyplot as plt 
							#plt.plot(self.chi_logspace_arr, ker_logspace, label='%d'%(len(self.t_s_logspace)))
							#plt.legend()
							#plt.show()							
							self.t_s_logspace.append(ker_logspace)

							# ker_logspace_interfunc = interp1d(np.log(self.chi_logspace_arr), ker_logspace, bounds_error=False, fill_value=0.)
							# self.ker_chi_interpfuncs.append(ker_logspace_interfunc)

						# plt.axvline(x=chi_min)
						# plt.axvline(x=chi_max)
						# plt.xscale('log')
						# plt.yscale('log')
						# plt.legend()
						# plt.show()
						# exit()

				else:
						print("calculator_fang.py: Nonlimber not supported!")
						sys.exit()
						nzs = self.get_tracer_dndzs()
						tpar = self.get_tracer_parameters()
						z_g = nzs['z_cl']
						z_s = nzs['z_sh']
						self.t_g = [ccl.NumberCountsTracer(self.cosmo, True,(z_g, nzs['dNdz_cl'][:, ni]),bias=(z_g,np.full(len(z_g), b)))
												for ni, b in zip(range(0, 10),tpar['b_g'])]
						self.t_s = [ccl.WeakLensingTracer(self.cosmo,(z_s, nzs['dNdz_sh'][:, ni]),True)
												for ni in range(0, 5)]
				self.fk_g = [None]*len(self.t_g)
				self.fk_s = [None]*len(self.t_s)
				#self.fk_g = [None]*len(self.t_g)
				#self.k_logspace_arr = [None]*len(self.t_g)

		def _get_cl_limber(self, t1, t2, ls):
				return ccl.angular_cl(self.cosmo, t1, t2, ls,
															limber_integration_method='qag_quad')
		#spline qag_quad
		def _get_cl_limber_nonlin_part(self, t1, t2, ls):
				return ccl.angular_cl(self.cosmo_nonlin_part, t1, t2, ls,limber_integration_method='qag_quad')
				# return ccl.angular_cl(self.cosmo, t1, t2, ls, limber_integration_method='qag_quad') - ccl.angular_cl(self.cosmo_nonlin_part, t1, t2, ls,limber_integration_method='qag_quad')


		def _get_cls_nonlimber_lin(self, fchi1, fchi2, ls, cl_type, i1, i2):
				# print(cl_type, i1, i2)
				nu = nu2= 1.01
				prefac = 1
				kpow = 3
				N_pad = Npad2 = 0#int(self.config['Npad_fft'])
				plaw=1.51
				plaw2 = 2.51
				plaw3 = 2.51
				# myfftlog1 = fftlog(self.chi_logspace_arr, fchi1, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=Npad)
				# myfftlog2 = fftlog(self.chi_logspace_arr, fchi2, nu=nu2, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=Npad2)
				Nell = ls.size
				cls_nonlimber_lin = np.zeros(Nell)
				chi_logspace_arr = []
				#if len(chi_logspace_arr) != len(self.chi_logspace_arr)+N_pad*2:
				chi_logspace_arr, fchi1 = self.pad_array(self.chi_logspace_arr, fchi1, N_pad)
				_, fchi2 = self.pad_array(self.chi_logspace_arr, fchi2, N_pad)
				#chi_logspace_arr = self.chi_logspace_arr
				#print(len(chi_logspace_arr), len(self.chi_logspace_arr),len(fchi1), (N_pad))

				# t0 = time.time()
				if(cl_type == "gg"):
					#if(self.fk_g[i1] is None):
					self.fk_g[i1]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))
					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi1, ell, plaw3, 1, 0.0, 0.0)
						self.k_logspace_arr[i], self.fk_g[i1][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk1 = self.fk_g[i1]
					#print(len(fk1), len(fk1[0]), len(self.t_g), len(self.t_s), Nell, len(self.chi_logspace_arr))
					#print(fk1[-1])

					#if(self.fk_g[i1+i2] is None):
					self.fk_g[i1+i2]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))
					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi2, ell, plaw3, 1, 0.0, 0.0)
						self.k_logspace_arr[i], self.fk_g[i1+i2][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk2 = self.fk_g[i1+i2]

				if(cl_type == "gs"):
					#if(self.fk_g[i1] is None):
					self.fk_g[i1]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))
					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi1, ell, plaw, 1, 0.0, 0.0)
						self.k_logspace_arr[i], self.fk_g[i1][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk1 = self.fk_g[i1]

					#if(self.fk_s[i2] is None):
					self.fk_s[i2]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))
					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi2, ell, plaw2, 1, 0.0, -2.0)
						self.k_logspace_arr[i], self.fk_s[i2][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk2 = self.fk_s[i2]
					
					prefac = (ls+2)*(ls+1)*(ls)*(ls-1)
					prefac[prefac<0] = 0.
					prefac = np.sqrt(prefac)

				if(cl_type == "ss"):
					prefac = (ls+2)*(ls+1)*(ls)*(ls-1)
					prefac[prefac<0] = 0.

					#if(self.fk_s[i1] is None):
					self.fk_s[i1]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))

					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi1, ell, plaw, 1, 0.0, -2.0)
						self.k_logspace_arr[i], self.fk_s[i1][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk1 = self.fk_s[i1]	

					#if(self.fk_s[i1+i2] is None):
					self.fk_s[i1+i2]=np.zeros((Nell,len(self.chi_logspace_arr)))
					self.k_logspace_arr=np.zeros((Nell,len(self.chi_logspace_arr)))

					for i in range(len(ls)):
						ell = ls[i]
						temp_k, temp_fk = _fftlog_transform_general(chi_logspace_arr, fchi2, ell, plaw, 1, 0.0, -2.0)
						self.k_logspace_arr[i], self.fk_s[i1+i2][i] = self.unpad_array(temp_k, temp_fk, N_pad)
					fk2 = self.fk_s[i1+i2]

				k = self.k_logspace_arr
				#print(len(k[0]),len(k))
				cls_nonlimber_lin = np.sum(fk1 * fk2 * k**kpow * np.exp(self.pklin_interp(np.log(k))), axis=1 ) * self.dlnr * 2./np.pi
				# t1 = time.time()
				# print(Nell, t1-t0)
				#plt.plot(ls, cls_nonlimber_lin*prefac)
				#plt.plot(ls, cls_nonlimber_lin)
				#plt.yscale('log')
				#tit = cl_type, i1, i2
				#plt.title(tit)
				#plt.show()
				chi_logspace_arr, fchi1 = self.unpad_array(chi_logspace_arr,fchi1,N_pad)
				_,fchi2 = self.unpad_array(chi_logspace_arr,fchi2,N_pad)
				#print(len(chi_logspace_arr), len(self.chi_logspace_arr),len(fchi1), (N_pad))

				return cls_nonlimber_lin*prefac

		def _get_cl_nonlimber_fang(self, t1, t2, ls, i1, i2, cl_type):

				l_switch = float(self.config['l_nonlimber'])
				if(cl_type=='gg'):
					ls_nonlim = ls[ls<l_switch]
					cls_limber_nonlin_part = self._get_cl_limber_nonlin_part(t1, t2, ls_nonlim)
					cls_nonlimber_lin = self._get_cls_nonlimber_lin(self.t_g_logspace[i1], self.t_g_logspace[i1+i2], ls_nonlim, cl_type, i1, i2)
					cls_nonlimber = cls_limber_nonlin_part + cls_nonlimber_lin
					cls_limber = self._get_cl_limber(t1, t2, ls[ls>=l_switch])
					return np.hstack((cls_nonlimber, cls_limber))
				# elif(cl_type=='gs'):
				# 	cls_nonlimber_lin =  self._get_cls_nonlimber_lin(self.t_g_logspace[i1], self.t_s_logspace[i2], ls_nonlim)
				# else:
				# 	cls_nonlimber_lin =  self._get_cls_nonlimber_lin(self.t_s_logspace[i1], self.t_s_logspace[i1+i2], ls_nonlim)

				if(cl_type=='gs'):
					ls_nonlim = ls[ls<l_switch]
					cls_limber_nonlin_part = self._get_cl_limber_nonlin_part(t1, t2, ls_nonlim)
					cls_nonlimber_lin = self._get_cls_nonlimber_lin(self.t_g_logspace[i1], self.t_s_logspace[i2], ls_nonlim, cl_type, i1, i2)
					cls_nonlimber = cls_limber_nonlin_part + cls_nonlimber_lin
					cls_limber = self._get_cl_limber(t1, t2, ls[ls>=l_switch])
					return np.hstack((cls_nonlimber, cls_limber))

				# elif(cl_type=='gs'):
				# 	cls_nonlimber_lin =  self._get_cls_nonlimber_lin(self.t_g_logspace[i1], self.t_s_logspace[i2], ls_nonlim)
				# else:
				# 	cls_nonlimber_lin =  self._get_cls_nonlimber_lin(self.t_s_logspace[i1], self.t_s_logspace[i1+i2], ls_nonlim)

				if(cl_type=='ss'):
					ls_nonlim = ls[ls<l_switch]
					cls_limber_nonlin_part = self._get_cl_limber_nonlin_part(t1, t2, ls_nonlim)
					cls_nonlimber_lin = self._get_cls_nonlimber_lin(self.t_s_logspace[i1], self.t_s_logspace[i1+i2], ls_nonlim, cl_type, i1, i2)
					cls_nonlimber = cls_limber_nonlin_part + cls_nonlimber_lin
					# cls_nonlimber = cls_limber_nonlin_part + cls_nonlimber_lin
					cls_limber = self._get_cl_limber(t1, t2, ls[ls>=l_switch])
					return np.hstack((cls_nonlimber, cls_limber))								
				else:
					return self._get_cl_limber(t1, t2, ls)
				# cls_nonlimber = cls_limber_nonlin_part + cls_nonlimber_lin
				# print(cls_nonlimber.size)
				# print(ls_nonlim.size)
				# exit()
				# cls_limber = self._get_cl_limber(t1, t2, ls[ls>=200])
				# return np.hstack((cls_nonlimber, cls_limber))

		def run(self):
				# Compute power spectra
				ls = self.get_ells()
				#ls=np.arange(2,100,dtype='double')

				self.cls_gg = []
				self.cls_gs = []
				self.cls_ss = []
				for i1, t1 in enumerate(self.t_g):
						for i2, t2 in enumerate(self.t_g[i1:]):
								self.cls_gg.append(self._get_cl_nonlimber_fang(t1, t2, ls, i1, i2, 'gg'))
						for i2, t2 in enumerate(self.t_s):
								self.cls_gs.append(self._get_cl_nonlimber_fang(t1, t2, ls, i1, i2, 'gs'))
				for i1, t1 in enumerate(self.t_s):
						for i2, t2 in enumerate(self.t_s[i1:]):
								self.cls_ss.append(self._get_cl_nonlimber_fang(t1, t2, ls, i1, i2, 'ss'))
				self.cls_gg = np.array(self.cls_gg)
				self.cls_gs = np.array(self.cls_gs)
				self.cls_ss = np.array(self.cls_ss)


def get_general_params(b):
	nu = 1.51
	nu2=2.51
	deriv = 0.0
	plaw = 0.0
	best_nu = nu
	if b<0: 
		plaw = -2.0
		best_nu = nu2

	if b<=0: deriv=0
	else: deriv = b

	return best_nu, plaw, deriv

def get_prefac(p, l):
	prefac = 1
	if (p==1 or p==2): prefac*=(l+2)*(l+1)*(l)*(l-1)
	if (prefac<=0): return 0
	if (p==1): prefac = np.sqrt(prefac)
	return prefac



def Implement_FKEM(cosmo, clt1, clt2, p_of_k_a, ls, l_limber, limber_max_error):
	#clt1, clt2 are lists of tracer in a tracer object
	#cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
	#psp non-linear power spectrum
	#l_limber max ell for non-limber calculation
	#ell_use ells at which we calculate the non-limber integrals


	kpow = 3

	#get relevant information from tracers to perform integrals
	#need chi_min and chi_max for each combination of tracers, choose most conservative range for limber integration
	#need to create log-spaced chi array given ranges
	#need to retieve pk lin and non-lin for each tracer
	#need kernels for each tracer
	#don't need to know what each tracer represents, need to retrieve bessel_deriv from the tracer.






	cells=np.zeros(len(ls))
	kernels_t1, chis_t1 = clt1.get_kernel()
	bessels_t1 = clt1.get_bessel_derivative()
	angles_t1 =  clt1.get_angles_derivative()
	kernels_t2, chis_t2 = clt2.get_kernel()
	bessels_t2 = clt2.get_bessel_derivative()
	angles_t2 =  clt2.get_angles_derivative()


	psp_lin = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
	psp_nonlin = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
	status=0
	cl_limber_lin, status = lib.angular_cl_vec_limber(cosmo, clt1, clt2, psp_lin, ls, 'qag_quad',ls.size, status)
	if status != 0: raise ValueError("Error in Limber integrator.")
	cl_limber_nonlin, status = lib.angular_cl_vec_limber(cosmo, clt1, clt2, psp_nonlin, ls, 'qag_quad',ls.size, status)
	if status != 0: raise ValueError("Error in Limber integrator.")
	pk = cosmo.get
	
	cells = []

	for el in range(len(ls)):
		ell = ls[el]
		cls_nonlimber_lin = 0.0
		for i in range(len(kernels_t1)):
			for j in range(len(kernels_t2)):
				#conservative min and max chis
				#this can be pre-computed to avoid repeated computations
				chi_min = np.max([np.min(chis_t1[i]), np.min(chis_t2[j])])
				chi_max = np.min([np.max(chis_t1[i]), np.max(chis_t2[j])])
				Nchi=np.min([len(chis_t1[i]), len(chis_t2[j])])
				chi_logspace_arr = np.logspace(np.log10(chi_min),np.log10(chi_max),num=Nchi, endpoint=True)
				dlnr = np.log(chi_max/chi_min)/(Nchi-1.)
				a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
				growfac_arr = ccl.growth_factor(cosmo, a_arr)

				fchi1_interp = interp1d(chis_t1[i], kernels_t1[i], fill_value='extrapolate')
				fchi2_interp = interp1d(chis_t2[j], kernels_t2[j], fill_value='extrapolate')
				fchi1 = fchi1_interp(chi_logspace_arr) * chi_logspace_arr*growfac_arr
				fchi2 = fchi2_interp(chi_logspace_arr) * chi_logspace_arr*growfac_arr



				nu, deriv, plaw = get_general_params(bessels_t1[i]) 
				k, fk1 = _fftlog_transform_general(chi_logspace_arr, fchi1, ell, nu, 1, deriv, plaw)
				nu, deriv, plaw = get_general_params(bessels_t2[j]) 
				k, fk2 = _fftlog_transform_general(chi_logspace_arr, fchi2, ell, nu, 1, deriv, plaw)
				prefac1, prefac2 = get_prefac(angles_t1[i], ell), get_prefac(angles_t2[j], ell)
				cls_nonlimber_lin += np.sum(fk1*prefac1 * fk2*prefac2 * k**kpow * psp_lin(k,1.0)) * dlnr * 2./np.pi


		cells.append(cl_limber_lin[el] - cl_limber_nonlin[el] + cls_nonlimber_lin)
		if np.abs(cl_limber_lin[el] - cl_limber_nonlin[el])<limber_max_error and l_limber=='auto':
			l_limber = ls[el]
			break


	return l_limber, cl_non_limber, status


