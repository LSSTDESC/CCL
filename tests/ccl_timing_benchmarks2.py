import numpy as np
import pyccl as ccl
import time

##### 
# This script tests the speed of CCL in a typical likelihood evaluation.

#####
# Tune the parameters below to explore different scenarios
#  a) Number of redshift bins (for both sources and lenses)
nbins=10
#  b) Type of cosmological model (1-> simple LCDM, 2-> LCDM with massive neutrinos, 3-> Use emulator to obtain P(k))
which_cosmo=1
#  c) Do sources have intrinsic alignments?
has_ia = True
#  d) Do lenses have redshift-space distortions?
#     This must be set to False for cosmologies with massive neutrinos
has_rsd = True
#  e) Do lenses have magnification bias?
has_mag = True

logyn=['no','yes']
print("Run parameters: ")
if which_cosmo==1 :
       print(" Vanilla LCDM")
elif which_cosmo==2 :
       print(" LCDM + m_nu")
elif which_cosmo==3 :
       print(" CosmicEmu")
else :
       raise ValueError("Please choose which_cosmo=1, 2 or 3")
print(" %d tomographic bins"%nbins)
print(" RSD  : "+logyn[int(has_rsd)])
print(" IAs  : "+logyn[int(has_ia)])
print(" Magnification : "+logyn[int(has_mag)])
print("")

def report_time(msg,tm) :
       print(msg+": %.5lf s"%tm)

#####
# A) The first step is to define the N(z)s of both lenses and sources.
#    For this particular example we use CCL's internal functions, but these are only useful for forecasting.
z = np.linspace(0.,2.0, 200)
pz_gaussian = ccl.PhotoZGaussian(0.05)
dNdz_lens_list=[]
dNdz_source_list=[]
for i in range(0,nbins):
       dNdz_source_list.append(ccl.dNdz_tomog(z, 'wl_fid', i*0.15, (i+1)*0.15, pz_gaussian))
       dNdz_lens_list.append(ccl.dNdz_tomog(z, 'nc', i*0.15, (i+1)*0.15, pz_gaussian))

#####
# B) Start the timer
t1=time.time()

#####
# C) Initialize the cosmological model
if which_cosmo==1 :
       cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
elif which_cosmo==2 :
       cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96,m_nu=0.06)
elif which_cosmo==3 :
       cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.7, sigma8=0.8, w0=-1., wa=0., n_s=0.96,Neff=3.04, transfer_function='emulator',matter_power_spectrum='emu')
t2=time.time()
report_time('Cosmology creation',t2-t1)

#####
# D) We first test how long it takes to compute the matter power spectrum from the chosen method.
s8=ccl.sigma8(cosmo)
t3=time.time()
report_time('P(k) call',t3-t2)

#####
# E) Now we will initialize one ClTracer object for each redshift bin of either lenses or sources.
#    First let us define some bias functions (galaxy bias, IA amplitudes, magnification bias)
bias_ia = np.ones(z.size) # Intrinsic alignment bias factor
f_red = 0.5 * np.ones(z.size) # Fraction of aligned galaxies
bias = np.ones(z.size) # Galaxy bias
mag_bias = np.ones(z.size) # Magnification bias
#    Now we actually generate the tracers
lenses=[]
sources=[]
for i in range(0,nbins):
       lenses.append(ccl.ClTracerLensing(cosmo, has_intrinsic_alignment=has_ia, z=z, n=dNdz_source_list[i], bias_ia=bias_ia, f_red=f_red))
       sources.append(ccl.ClTracerNumberCounts(cosmo, has_rsd=has_rsd, has_magnification=has_mag,n=dNdz_lens_list[i], bias=bias, mag_bias=mag_bias, z=z))
t4=time.time()
report_time('Tracer creation',t4-t3)

#####
# F) Now compute all possible cross-power spectra
#    ell-range chosen to give good coverage to compute correlation function later on
ell = np.arange(1,10000)
nell=len(ell)
#    Galaxy-galaxy power spectra
cl_gg=np.zeros([nbins,nbins,nell])
for i in range(0,nbins) :
       for j in range(i,nbins) :
              cl_gg[i,j,:]=ccl.angular_cl(cosmo,sources[i],sources[j],ell,l_linstep=10000,l_logstep=1.1)
              if j!=i :
                     cl_gg[j,i,:]=cl_gg[i,j,:]
#    Galaxy-lensing power spectra
cl_gl=np.zeros([nbins,nbins,nell])
for i in range(0,nbins) :
       for j in range(0,nbins) :
              cl_gl[i,j,:]=ccl.angular_cl(cosmo,sources[i],lenses[j],ell,l_linstep=10000,l_logstep=1.1)
#    Lensing-lensing power spectra
cl_ll=np.zeros([nbins,nbins,nell])
for i in range(0,nbins) :
       for j in range(i,nbins) :
              cl_ll[i,j,:]=ccl.angular_cl(cosmo,lenses[i],lenses[j],ell,l_linstep=10000,l_logstep=1.1)
              if j!=i :
                     cl_ll[j,i,:]=cl_ll[i,j,:]
#    Total number of power spectra computed
total_pspec=nbins*(nbins+1)+nbins**2
t5=time.time()
report_time('%d Cls'%total_pspec,t5-t4)

#####
# G) Finally, we compute the angular correlation functions
#    Generic angular range (in degrees)
theta_deg = np.logspace(-1, np.log10(5.), 20)
nth=len(theta_deg)
corrmethod='FFTLog'
#    Galaxy-galaxy (w)
xi_gg=np.zeros([nbins,nbins,nth])
for i in range(0,nbins) :
       for j in range(i,nbins) :
              xi_gg[i,j,:]=ccl.correlation(cosmo,ell,cl_gg[i,j],theta_deg,corr_type='GG',method=corrmethod)
              if j!=i :
                     xi_gg[j,i,:]=xi_gg[i,j,:]
#    Galaxy-lensing (gamma_T)
xi_gl=np.zeros([nbins,nbins,nth])
for i in range(0,nbins) :
       for j in range(0,nbins) :
              xi_gl[i,j,:]=ccl.correlation(cosmo,ell,cl_gl[i,j],theta_deg,corr_type='GL',method=corrmethod)
              if j!=i :
                     xi_gl[j,i,:]=xi_gl[i,j,:]
#    Lensing-lensing (xi_+, xi_-)
xi_p=np.zeros([nbins,nbins,nth])
xi_m=np.zeros([nbins,nbins,nth])
for i in range(0,nbins) :
       for j in range(i,nbins) :
              xi_p[i,j,:]=ccl.correlation(cosmo,ell,cl_ll[i,j],theta_deg,corr_type='L+',method=corrmethod)
              xi_m[i,j,:]=ccl.correlation(cosmo,ell,cl_ll[i,j],theta_deg,corr_type='L-',method=corrmethod)
              if j!=i :
                     xi_p[j,i,:]=xi_p[i,j,:]
                     xi_m[j,i,:]=xi_m[i,j,:]
#    Total number of correlation functions computed
total_corrs=3*((nbins*(nbins+1))/2)+nbins**2
t6=time.time()
report_time('%d correlation functions'%total_corrs,t6-t5)

#Final reckoning
report_time("Total time ellapsed",t6-t1)
