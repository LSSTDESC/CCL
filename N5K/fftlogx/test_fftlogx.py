"""
This module tests the python wrapper for cfftlog
The tests contain:
-- integrals with 1 spherical Bessel function;
-- integrals with 1 1st-derivative of spherical Bessel function;
-- integrals with 1 2nd-derivative of spherical Bessel function;
-- integrals with 1 (cylindrical) Bessel function, i.e. Hankel transform;

by Xiao Fang
Oct 25, 2021
"""

import numpy as np
from fftlogx import *
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform_general
from pyccl.pyutils import _fftlog_transform

def window_arr(k):
	k_min = np.min(k)
	k_max = np,max(k)
	k_left = 0
	k_right = 1
	window = []
	for i in range(len(k)):
		if k[i] < k_left:
			frac = (k[i]-k_min)/(k_left - k_min)
			frac -= 1/(2*np.pi)*np.sin(2*np.pi*frac)
			window.append(frac)
		elif k[i] > k_right:
			frac = (k_max - k[i])/(k_max - k_right)
			frac -= 1/(2*np.pi)*np.sin(2*np.pi*frac)
			window.append(frac)
		else:
			window.append(1)
	return window




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

h=0.6727
cosmo_fid_srd = ccl.Cosmology(
    Omega_c=0.2664315,
    Omega_b=0.0491685,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
    sigma8 = 0.831,
    n_s=0.9645,
    h=h,
)

cosmo_fid_srd.compute_nonlin_power()
print('This is a test of fftlog module written by Xiao Fang.')
print('nu is required to be between -ell to 2.')
k, pk = np.loadtxt('./Pk_test', usecols=(0,1), unpack=True)
k = np.logspace(-5, 2, 2000)
pk = ccl.power.linear_power(cosmo_fid_srd, k, 1)
#pk *=k
N = k.size
print('number of input data points: '+str(N))
ell = 52.
nu = 1.01
nu_2 = 1.01
#pk = -(np.log(k)-1)**2 + 10**5
#myfftlog = fftlog(k, pk, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)
myfftlog = fftlog(k, pk, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=200)
myfftlog2 = fftlog(k, pk, nu=nu_2, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=200)


N_low = int(0)#*(ell+1))
N_high =int(0)#*(ell+1))
#pk = np.power(k,-1)
k = log_extrap(k, N_low,N_high)
pk = log_extrap(pk, N_low, N_high)
#pk = np.ones(len(k))

N = len(k)
'''
N_pad=int(200)#*(ell+1))
if(N_pad):
	pad = np.zeros(N_pad)
	k = log_extrap(k, N_pad, N_pad)
	pk = np.hstack((pad, pk, pad))
	N_low += N_pad
	N_high += N_pad
	N+=2*N_pad

if(len(pk)%2==1): # Make sure the array sizes are even
	k= k[:-1]
	pk=pk[:-1]
	N-=1
	if(N_pad):
		N_high -=1
'''
print(len(pk))
#print(k)
smooth_scaled_pk =pk#/k#/k**nu#/k**1.01#/k#/(2*np.pi)**4

def low_ring_q(ell):
	return 1.6/(ell+1) - 0.8

def low_ring_q_plaw(ell):
	return abs(low_ring_q(ell))

#np.array([0.]),np.array([0.])
plaw = -0.1#1.01
plaw1 = .1#1.01
plaw2 = .71#1.01
plaw3 = .71
#ccl_r_old, ccl_Fr_old = _fftlog_transform(k, smooth_scaled_pk, 2, ell+0.5, -1.0)
ccl_r, ccl_Fr = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw, 1, 0.0, 0.0)
ccl_r1, ccl_Fr1 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw1, 1, 1.0, 0.0)
ccl_r2, ccl_Fr2 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw2, 1, 2.0, 0.0)
ccl_r3, ccl_Fr3 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw3, 1, 0.0, -2.0)

plaw_2 = -np.log10(ell+1.01)/np.log10(100)#-0.51/2
print(np.log10(ell)/np.log10(100))
plaw_2 = low_ring_q(ell)
plaw1_2 = low_ring_q(ell)#-.8+1.e-8
plaw2_2 = low_ring_q(ell)#-.8+1.e-8
plaw3_2 = low_ring_q_plaw(ell)/4#-.8 + 1.e-8
print("low_ring",plaw3_2)
ccl_r_2, ccl_Fr_2 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw_2, 1, 0.0, 0.0)
ccl_r1_2, ccl_Fr1_2 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw1_2, 1, 1.0, 0.0)
ccl_r2_2, ccl_Fr2_2 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw2_2, 1, 2.0, 0.0)
ccl_r2_3, ccl_Fr2_3 = _fftlog_transform_general(k, smooth_scaled_pk, ell, plaw3_2, 1, 0.0, -2.0)


#ccl_r_3d, ccl_Fr_3d = np.array([0.]),np.array([0.])#_fftlog_transform(k, smooth_scaled_pk, 3, ell,.5)
epsilon = 0 + plaw

r, Fr = myfftlog.fftlog(ell)
r_2, Fr_2 = myfftlog2.fftlog(ell)
r1, Fr1 = myfftlog.fftlog_dj(ell)
r1_2, Fr1_2 = myfftlog2.fftlog_dj(ell)
r2, Fr2 = myfftlog.fftlog(ell)
r2_2, Fr2_2 = myfftlog2.fftlog_ddj(ell)
r2_3, Fr2_3 = myfftlog2.fftlog_dj_modified_ells(np.array([ell]))
r3_3, Fr3_3 = myfftlog2.fftlog_modified_ells(np.array([ell]))
print(len(ccl_r))
'''
k = k[(N_low):]
k = k[:(-N_high)]
pk = pk[(N_low):]
pk= pk[:(-N_high)]

ccl_r = ccl_r[(N_low):]
ccl_r = ccl_r[:(-N_high)]
ccl_Fr = ccl_Fr[(N_low):]
ccl_Fr = ccl_Fr[:(-N_high)]

ccl_r1 = ccl_r1[(N_low):]
ccl_r1 = ccl_r1[:(-N_high)]
ccl_Fr1 = ccl_Fr1[(N_low):]
ccl_Fr1 = ccl_Fr1[:(-N_high)]

ccl_r2 = ccl_r2[(N_low):]
ccl_r2 = ccl_r2[:(-N_high)]
ccl_Fr2 = ccl_Fr2[(N_low):]
ccl_Fr2 = ccl_Fr2[:(-N_high)]


ccl_r3 = ccl_r3[(N_low):]
ccl_r3 = ccl_r3[:(-N_high)]
ccl_Fr3 = ccl_Fr3[(N_low):]
ccl_Fr3 = ccl_Fr3[:(-N_high)]


ccl_r_2 = ccl_r_2[(N_low):]
ccl_r_2 = ccl_r_2[:(-N_high)]
ccl_Fr_2 = ccl_Fr_2[(N_low):]
ccl_Fr_2 = ccl_Fr_2[:(-N_high)]

ccl_r1_2 = ccl_r1_2[(N_low):]
ccl_r1_2 = ccl_r1_2[:(-N_high)]
ccl_Fr1_2 = ccl_Fr1_2[(N_low):]
ccl_Fr1_2 = ccl_Fr1_2[:(-N_high)]

ccl_r2_2 = ccl_r2_2[(N_low):]
ccl_r2_2 = ccl_r2_2[:(-N_high)]
ccl_Fr2_2 = ccl_Fr2_2[(N_low):]
ccl_Fr2_2 = ccl_Fr2_2[:(-N_high)]

ccl_r2_3 = ccl_r2_3[(N_low):]
ccl_r2_3 = ccl_r2_3[:(-N_high)]
ccl_Fr2_3 = ccl_Fr2_3[(N_low):]
ccl_Fr2_3 = ccl_Fr2_3[:(-N_high)]
'''
'''
for i in range(len(ccl_r)-1- N_low,0):
	#ccl_Fr[i] *=ccl_r[i]**plaw 
	ccl_r[i] = ccl_r[i+N_low]
	ccl_Fr[i] = ccl_Fr[i+N_low]
	#ccl_Fr[i] /= ccl_r[i]**plaw

ccl_r = ccl_r[:(-N_high)]
ccl_Fr = ccl_Fr[:(-N_high)]
'''
r_scale_factor = r[0]/ccl_r[0]

#ccl_Fr1 *=ccl_r1**plaw1


#for i in range(len(ccl_r)):
#	ccl_r1[i] = ccl_r1[i] *r_scale_factor#/(2*np.pi*np.pi*np.exp(1)/55)#*k[0]*np.exp(-i/N)/k[N-i-1]

#ccl_r *= (ell+1.)



#ccl_Fr*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr1*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr2*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr_2*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr1_2*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr2_2*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr3*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr2_3*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))
#ccl_Fr1=np.abs(ccl_Fr1)/2/2/2/2
#ccl_Fr/=(1+ell)**0.01*(N-1)/(N)
#ccl_Fr1/=ccl_r1**nu
#ccl_Fr_2*=ccl_r_2**(-0.2)
#ccl_Fr_2/=26.97870130018149 
#ccl_Fr_2*=np.sqrt(np.pi)/4#*np.exp((len(k)-1)/len(k))


print(np.max(ccl_Fr), np.max(Fr), np.max(Fr)/np.max(ccl_Fr))
print(ccl_Fr[-1], Fr[-1], Fr[-1]/ccl_Fr[-1])
print(r[0], r[-1], ccl_r[0], ccl_r[-1], r_scale_factor, r[-1]/ccl_r[-1])
#print(np.max(ccl_Fr_2), np.max(Fr_2), np.max(Fr_2)/np.max(ccl_Fr_2))
#print(ccl_Fr_2[-1], Fr_2[-1], Fr_2[-1]/ccl_Fr_2[-1])
#print(r_2[0], r_2[-1], ccl_r_2[0], ccl_r_2[-1], r_2[-1]/ccl_r_2[-1])

#print('indexes: ', np.argmax(Fr_2), np.argmax(ccl_Fr_2), np.argmax(ccl_Fr_2) - np.argmax(Fr_2))
#print('indexes: ', np.argmax(Fr1_2), np.argmax(ccl_Fr1_2), np.argmax(ccl_Fr1_2) - np.argmax(Fr1_2))
#print('indexes: ', np.argmax(Fr2_2), np.argmax(ccl_Fr2_2), np.argmax(ccl_Fr2_2) - np.argmax(Fr2_2))

'''
print(np.max(ccl_Fr1), np.max(Fr1), np.max(Fr1)/np.max(ccl_Fr1))
print(ccl_Fr1[-1], Fr1[-1], Fr1[-1]/ccl_Fr1[-1])
print(r1[0], r1[-1], ccl_r1[0], ccl_r1[-1], r1[0]/ccl_r1[0], r1[-1]/ccl_r1[-1])
print(np.max(ccl_Fr2), np.max(Fr2), np.max(Fr2)/np.max(ccl_Fr2))
print(ccl_Fr2[-1], Fr2[-1], Fr2[-1]/ccl_Fr2[-1])
print(r2[0], r2[-1], ccl_r2[0], ccl_r2[-1], r2[0]/ccl_r2[0], r2[-1]/ccl_r2[-1])

print(np.max(ccl_Fr1_2), np.max(Fr1_2), np.max(Fr1_2)/np.max(ccl_Fr1_2))
print(ccl_Fr1_2[-1], Fr1_2[-1], Fr1_2[-1]/ccl_Fr1_2[-1])
print(r1_2[0], r1_2[-1], ccl_r1_2[0], ccl_r1_2[-1], r1_2[0]/ccl_r1_2[0], r1_2[-1]/ccl_r1_2[-1])
print(np.max(ccl_Fr2_2), np.max(Fr2_2), np.max(Fr2_2)/np.max(ccl_Fr2_2))
print(ccl_Fr2_2[-1], Fr2_2[-1], Fr2_2[-1]/ccl_Fr2_2[-1])
print(r2_2[0], r2_2[-1], ccl_r2_2[0], ccl_r2_2[-1], r2_2[0]/ccl_r2_2[0], r2_2[-1]/ccl_r2_2[-1])
'''

for i in range(len(ccl_r)):
	if ccl_r3[i]>r2_3[0][0]: 
		print(i)
		break

for i in range(len(ccl_r)):
	if ccl_r3[i]>r2_3[0][-1]: 
		print(i)
		break
r_min = 25

print(ccl_r_2[r_min + np.argmax(ccl_Fr_2[r_min:i])], r_2[np.argmax(Fr_2)], ccl_r_2[r_min + np.argmax(ccl_Fr_2[r_min:i])]/r_2[np.argmax(Fr_2)])
print(ccl_Fr_2[r_min + np.argmax(ccl_Fr_2[r_min:i])], Fr_2[np.argmax(Fr_2)], ccl_Fr_2[r_min + np.argmax(ccl_Fr_2[r_min:i])]/Fr_2[np.argmax(Fr_2)], 2**(ell+1))
print(ccl_r[r_min + np.argmax(ccl_Fr[r_min:i])], r[np.argmax(Fr)], ccl_r[r_min + np.argmax(ccl_Fr[r_min:i])]/r[np.argmax(Fr)])
print(ccl_Fr[r_min + np.argmax(ccl_Fr[r_min:i])], Fr[np.argmax(Fr)], ccl_Fr[r_min + np.argmax(ccl_Fr[r_min:i])]/Fr[np.argmax(Fr)])
print(np.max(ccl_Fr[r_min:i]), np.max(ccl_Fr_2[r_min:i]), np.max(ccl_Fr_2[r_min:i])/np.max(ccl_Fr[r_min:i]))
print(ccl_r[r_min + np.argmax(ccl_Fr[r_min:i])], ccl_r_2[r_min + np.argmax(ccl_Fr_2[r_min:i])], ccl_r[r_min + np.argmax(ccl_Fr[r_min:i])]/ccl_r_2[r_min + np.argmax(ccl_Fr_2[r_min:i])])

################# Test fftlog ##############
print('Testing fftlog')
fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^{\infty} f(x)j_{\ell}(xy) dx/x, \ell=$%.1f'%(ell))

subfig1 = fig.add_subplot(1,5,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
#plt.tight_layout()

subfig2 = fig.add_subplot(1,5,2)
temp = r'$\nu=$'+str(nu)+r'or $\nu=$'+str(1.e-8)
subfig2.set_title(temp)
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
#subfig2.set_xlim( np.min(r), np.max(r))
#subfig2.set_ylim(10**-9, np.max(Fr))



subfig2.plot(ccl_r, ccl_Fr, label='fftlog from CCL', alpha=0.5)
subfig2.plot(ccl_r_2, ccl_Fr_2, label='fftlog from CCL low bias index')
#subfig2.plot(r, Fr, label='fftlogx')
#subfig2.plot(ccl_r_old, ccl_Fr_old, label='fftlog from CCL old')

subfig2.plot(r_2, Fr_2, label='fftlogx low bias index',alpha=0.5)

subfig2.legend()


subfig3 = fig.add_subplot(1,5,3)
#subfig3.set_title(r'$F(y) = \int_0^{\infty} f(x)J_{\ell}(xy) dx/x, \ell=$%.1f'%(ell))
subfig3.set_xscale('log')
subfig3.set_yscale('log')
subfig3.set_xlabel('y')
subfig3.set_ylabel('F(y)')
#subfig3.set_xlim( np.min(r1_2), np.max(r1_2))
subfig3.set_ylim(10**-9, np.max(Fr1))

#subfig3.set_ylim(10**-2, np.max(Fr))
#subfig3.plot(r, ccl_r, label='fftlog from CCL')
#subfig3.plot(r1, Fr1, label='fftlogx')

subfig3.plot(ccl_r1, ccl_Fr1, label='fftlog from CCL', alpha=0.5)
subfig3.plot(ccl_r1_2, ccl_Fr1_2, label='fftlog from CCL low bias index')
subfig3.plot(r1_2, Fr1_2, label='fftlogx low bias index', alpha=0.5)

subfig3.legend()


subfig4 = fig.add_subplot(1,5,4)
#subfig3.set_title(r'$F(y) = \int_0^{\infty} f(x)J_{\ell}(xy) dx/x, \ell=$%.1f'%(ell))
subfig4.set_xscale('log')
subfig4.set_yscale('log')
subfig4.set_xlabel('y')
subfig4.set_ylabel('F(y)')
#subfig4.set_xlim( np.min(r2_2), np.max(r2_2))
#subfig4.set_ylim(10**-9, np.max(Fr2))

#subfig4.plot(r2, Fr2, label='fftlogx')

subfig4.plot(ccl_r2, ccl_Fr2, label='fftlog from CCL', alpha=0.5)
subfig4.plot(ccl_r2_2, ccl_Fr2_2, label='fftlog from CCL low bias index')
subfig4.plot(r2_2, Fr2_2, label='fftlogx low bias index', alpha=0.5)



subfig5 = fig.add_subplot(1,5,5)
#subfig3.set_title(r'$F(y) = \int_0^{\infty} f(x)J_{\ell}(xy) dx/x, \ell=$%.1f'%(ell))
subfig5.set_xscale('log')
subfig5.set_yscale('log')
subfig5.set_xlabel('y')
subfig5.set_ylabel('F(y)')
#subfig4.set_xlim( np.min(r2_3[0]), np.max(r2_3[0]))
print( np.min(r2_3[0]), np.max(r2_3[0]))
#subfig4.set_ylim(10**-9, np.max(Fr2))

#subfig4.plot(r2, Fr2, label='fftlogx')

subfig5.plot(ccl_r3, ccl_Fr3, label='fftlog from CCL', alpha=0.5)
subfig5.plot(ccl_r2_3, ccl_Fr2_3, label='fftlog from CCL low bias index')
subfig5.plot(r2_3[0], Fr2_3[0], label='fftlogx low bias index', alpha=0.5)

subfig5.plot(r3_3[0], Fr3_3[0], label='fftlogx low bias index original', alpha=0.5)

'''
L = np.log(k[N-1]/k[0])/(N-1)*N
temp_k = []
for i in range(len(k)):
	temp_k.append(k[0]*np.exp(i*L/N))


subfig4.plot(k, temp_k/k - 1)
'''
#subfig4.legend()# r_c, Fr_c = np.loadtxt('../cfftlog/test_output.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_c, Fr_c, label='(bad) brute-force')

# r_bf, Fr_bf = np.loadtxt('test_bruteforce.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_bf, Fr_bf)
plt.legend()
#plt.tight_layout()
plt.show()

################# Test j' ##############
'''
print('Testing 1st & 2nd-derivative')

r1, Fr1 = myfftlog.fftlog_dj(ell)
r2, Fr2 = myfftlog.fftlog_ddj(ell)
fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^{\infty} f(x)j_{\ell}^{(n)}(xy) dx/x, \ell=$%.1f, n=1,2'%(ell))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('|F(y)|')
subfig2.plot(r1, abs(Fr1), label="1st-derivative")
subfig2.plot(r2, abs(Fr2), '--', label='2nd-derivative')
#r_bf, Fr_bf = np.loadtxt('test_bruteforce.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_bf, Fr_bf)
plt.legend()
plt.tight_layout()
plt.show()

################# Test Hankel ##############
print('Testing hankel')

n = 0
nu = 1.01
myhankel = hankel(k, pk, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.)
r, Fr = myhankel.hankel(n)

fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^{\infty} f(x)J_{n}(xy) dx/x, n=$%d'%(n))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
subfig2.plot(r, Fr)
plt.tight_layout()
plt.show()
'''


print('Testing non-spherical bessel integration')
plaw = 1.01
epsilon = 1.01
dim=2
ccl_r, ccl_Fr = _fftlog_transform_general(k, smooth_scaled_pk*(k**dim)*k**(-epsilon), ell, plaw, 0, 0.0, 0.0)
ccl_r_fid, ccl_Fr_fid = _fftlog_transform(k, smooth_scaled_pk, 2, ell, epsilon)
#ccl_Fr/=(4/np.sqrt(np.pi))
ccl_Fr/=((ell+1)**(1-epsilon))
#ccl_Fr_fid*=(2*np.pi)
ccl_Fr_fid*=ccl_r_fid**(epsilon)*(2*np.pi)**(dim/2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(ccl_r, ccl_Fr, label='general fftlog')
plt.plot(ccl_r_fid, ccl_Fr_fid, label='original fftlog')
plt.legend()
plt.show()

plaw = 1.01
epsilon = 0.01
dim=3
ccl_r, ccl_Fr = _fftlog_transform_general(k, smooth_scaled_pk*(k**2)*k**(-epsilon), ell, plaw, 1, 0.0, 0.0)
ccl_r_fid, ccl_Fr_fid = _fftlog_transform(k, smooth_scaled_pk, 3, ell, epsilon)
#ccl_Fr/=(4/np.sqrt(np.pi))
ccl_Fr/=((ell+1)**(1.5-epsilon))
ccl_Fr*=(2*np.pi)**2*((4/np.sqrt(np.pi))/2)**2
#*(2*np.pi)**(dim/2))
ccl_Fr_fid*=ccl_r_fid**(1+epsilon)/(2*np.pi**2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(ccl_r, ccl_Fr, label='general fftlog')
plt.plot(ccl_r_fid, ccl_Fr_fid, label='original fftlog')
plt.legend()
plt.show()