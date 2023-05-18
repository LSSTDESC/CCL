__all__ = ("implement_FKEM",)

import warnings

import numpy as np

from . import DEFAULT_POWER_SPECTRUM, CCLWarning, check, lib, warn_api
from .pyutils import integ_types

from scipy.interpolate import interp1d
#from fftlogx.fftlogx import *
#from pyccl.pyutils import _fftlog_transform
import time
import sys
from pyccl.pyutils import _fftlog_transform_general
import matplotlib.pyplot as plt



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

#incorrect!
def get_average_chi(a_arr, kernel):
	assert(len(a_arr)==len(kernel))
	new_arr = []
	for i in range(len(kernel)):
		if kernel[i]!=0:
			return i

	new_arr = np.array(new_arr)
	return new_arr.mean()

def get_kernel_min_max_index( kernel):
	mini = 0
	maxi = -1
	for i in range(len(kernel)):
		if kernel[i]!=0:
			mini=i
			break

	for i in range(len(kernel)-1,0):
		if kernel[i]!=0:
			maxi=i+1
			break
	if (maxi==len(kernel)-1): maxi=-1
	return mini,maxi

def implement_FKEM(cosmo,clt1, clt2, p_of_k_a, ls, l_limber, limber_max_error):
	import pyccl as ccl
	#clt1, clt2 are lists of tracer in a tracer object
	#cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
	#psp non-linear power spectrum
	#l_limber max ell for non-limber calculation
	#ell_use ells at which we calculate the non-limber integrals


	kpow = 3
	k_low = 1.e-3

	cells=np.zeros(len(ls))
	kernels_t1, chis_t1 = clt1.get_kernel()#need to trim non-zero values
	bessels_t1 = clt1.get_bessel_derivative()
	angles_t1 =  clt1.get_angles_derivative()
	kernels_t2, chis_t2 = clt2.get_kernel()#need to trim non-zero values
	bessels_t2 = clt2.get_bessel_derivative()
	angles_t2 =  clt2.get_angles_derivative()


	#l_limber='auto'


	fll_t1 = clt1.get_f_ell(ls)
	fll_t2 = clt2.get_f_ell(ls)


	psp_lin = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
	psp_nonlin = cosmo.parse_pk2d(p_of_k_a, is_linear=False)

	status = 0
	t1, status = lib.cl_tracer_collection_t_new(status)
	t2, status = lib.cl_tracer_collection_t_new(status)
	for t in clt1._trc:
	    status = lib.add_cl_tracer_to_collection(t1, t, status)
	for t in clt2._trc:
	    status = lib.add_cl_tracer_to_collection(t2, t, status)
	pk = cosmo.get_linear_power(name=p_of_k_a)
	pk_non = cosmo.get_nonlin_power(name=p_of_k_a)


	status=0
	cl_limber_lin, status = lib.angular_cl_vec_limber(cosmo.cosmo, t1, t2, psp_lin, ls, integ_types['qag_quad'],ls.size, status)
	if status != 0: raise ValueError("Error in Limber integrator.")
	cl_limber_nonlin, status = lib.angular_cl_vec_limber(cosmo.cosmo, t1, t2, psp_nonlin, ls, integ_types['qag_quad'],ls.size, status)
	if status != 0: raise ValueError("Error in Limber integrator.")
	
	
	cells = []
	for el in range(len(ls)):
		ell = ls[el]
		cls_nonlimber_lin = 0.0
		#cls_nonlimber_lin_temp = 0.0
		for i in range(len(kernels_t1)):
			for j in range(len(kernels_t2)):
				#mini_t1, maxi_t1 = get_kernel_min_max_index(kernels_t1[i])
				#mini_t2, maxi_t2 = get_kernel_min_max_index(kernels_t2[j])
				#kernels_t1[i] = kernels_t1[i][mini_t1:maxi_t1]
				#kernels_t2[j] = kernels_t2[j][mini_t2:maxi_t2]
				#chis_t1[i] = chis_t1[i][mini_t1:maxi_t1]
				#chis_t2[j] = chis_t2[j][mini_t2:maxi_t2]
				#conservative min and max chis
				#this can be pre-computed to avoid repeated computations
				chi_min = np.max([np.min(chis_t1[i]), np.min(chis_t2[j])])
				chi_max = np.min([np.max(chis_t1[i]), np.max(chis_t2[j])])
				Nchi=np.min([len(chis_t1[i]), len(chis_t2[j])])

				if chi_min==0.0: chi_min=0.0000001#get_chi_min_index(kernels_t1[i])
				chi_logspace_arr = np.logspace(np.log10(chi_min),np.log10(chi_max),num=Nchi, endpoint=True)
				dlnr = np.log(chi_max/chi_min)/(Nchi-1.)
				a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
				growfac_arr = ccl.growth_factor(cosmo, a_arr)

				transfer_t1 = np.array(clt1.get_transfer(np.log10(k_low), a_arr))
				#print(np.shape(transfer_t1))
				avg_chi1_i = get_average_chi(a_arr, kernels_t1[i])
				avg_chi2_i = get_average_chi(a_arr, kernels_t2[i])
				transfer_t2 = np.array(clt2.get_transfer(np.log10(k_low), a_arr))
				transfer_t1_avg = clt1.get_transfer(np.log10(k_low), a_arr[avg_chi1_i])
				transfer_t2_avg = clt1.get_transfer(np.log10(k_low), a_arr[avg_chi2_i])

				#print(avg_chi2_i, avg_chi2_i, transfer_t1_avg[i], transfer_t2_avg[j], transfer_t1[i], transfer_t2[j])

				fchi1_interp = interp1d(chis_t1[i], kernels_t1[i], fill_value='extrapolate')
				fchi2_interp = interp1d(chis_t2[j], kernels_t2[j], fill_value='extrapolate')
				fchi1 = fchi1_interp(chi_logspace_arr) * chi_logspace_arr*growfac_arr * transfer_t1[i]/transfer_t1_avg[i]
				fchi2 = fchi2_interp(chi_logspace_arr) * chi_logspace_arr*growfac_arr* transfer_t2[j]/transfer_t2_avg[j]
				#print(chi_logspace_arr[avg_chi1_i])
				#print(chi_logspace_arr)
				#plt.loglog(chi_logspace_arr, fchi1_interp(chi_logspace_arr))
				#plt.show()

				#print(np.shape(kernels_t1[i]), np.shape(chis_t1[i]))
				nu, deriv, plaw = get_general_params(bessels_t1[i]) 
				k, fk1 = _fftlog_transform_general(chi_logspace_arr, fchi1, float(ell), nu, 1, float(deriv), float(plaw))
				nu, deriv, plaw = get_general_params(bessels_t2[j]) 
				#print(nu, deriv, plaw)
				k, fk2 = _fftlog_transform_general(chi_logspace_arr, fchi2, float(ell), nu, 1, float(deriv), float(plaw))
				transfer_t1 = np.array(clt1.get_transfer(k, a_arr[avg_chi1_i]))[i]
				transfer_t2 = np.array(clt2.get_transfer(k, a_arr[avg_chi2_i]))[j]
				#print(transfer_t1, transfer_t2)
				#prefac1, prefac2 = get_prefac(angles_t1[i], ell), get_prefac(angles_t2[j], ell)
				cls_nonlimber_lin +=np.sum(fk1 * transfer_t1* fk2 * transfer_t2 * k**kpow * pk(k,1.0, cosmo)) * dlnr * 2./np.pi
				#cls_nonlimber_lin_temp +=np.sum(fk1 * fk2  * k**kpow * pk(k,1.0)) * dlnr * 2./np.pi
				#print(cls_nonlimber_lin, i, j)

		#print(fll_t1[i][el]*fll_t2[j][el], len(cl_limber_nonlin),len(ls), cls_nonlimber_lin, cl_limber_nonlin[el], cl_limber_nonlin[el]- cl_limber_lin[el])
		cells.append(cl_limber_nonlin[el]- cl_limber_lin[el] + fll_t1[i][el]*fll_t2[j][el]*cls_nonlimber_lin)
		if (cells[-1]/cl_limber_nonlin[el] - 1<limber_max_error and l_limber=='auto') or (type(l_limber) !='str' and  ell >=l_limber):
			l_limber = ls[el]
			break


	return l_limber, cells, status