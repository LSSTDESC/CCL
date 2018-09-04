import numpy as np
import matplotlib.pyplot as plt

#Generates redshift distribution for angular power spectrum and correlation function benchmarks
#Contact david.alonso@physics.ox.ac.uk if you have issues running this script

nz=256
sz=0.15
z1=1.0
z2=1.5
z1_arr=z1-np.fmin(20*sz,z1)+40*sz*(np.arange(nz)+0.5)/nz
z2_arr=z2-np.fmin(20*sz,z2)+40*sz*(np.arange(nz)+0.5)/nz
pz1_arr=np.exp(-0.5*((z1_arr-z1)/sz)**2)
pz2_arr=np.exp(-0.5*((z2_arr-z2)/sz)**2)

zlo_arr,h1,h2=np.loadtxt("z_DESC-CC",unpack=True)
dz=np.mean(zlo_arr[1:]-zlo_arr[:-1])
zm_arr=zlo_arr+dz/2

z_arr_bias=1.25*(np.arange(2*nz)+0.5)/nz
bias_arr=np.ones(len(z_arr_bias))


plt.plot(z1_arr,pz1_arr)
plt.plot(z2_arr,pz2_arr)
plt.show()

plt.plot(zm_arr,h1)
plt.plot(zm_arr,h2)
plt.show()

plt.plot(z_arr_bias,bias_arr)
plt.show()


np.savetxt("bin1_analytic.txt",np.transpose([z1_arr,pz1_arr]))
np.savetxt("bin2_analytic.txt",np.transpose([z2_arr,pz2_arr]))

np.savetxt("bin1_histo.txt",np.transpose([zm_arr,h1]))
np.savetxt("bin2_histo.txt",np.transpose([zm_arr,h2]))

np.savetxt("bias.txt",np.transpose([z_arr_bias,bias_arr]))
