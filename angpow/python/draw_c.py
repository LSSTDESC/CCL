
# utlity for a quick look at the Cl and c(theta) provided by Angpow
# input should be a path to the file **without the "cl.txt" or "ctheta.txt" suffix
# author: S. Plaszczynski, J. Neveu, 09feb2017

#print backend
import matplotlib
print(matplotlib.get_backend())
#matplotlib.use("TkAgg") 


import numpy as np
import sys, os
from matplotlib import pyplot as plt


plt.figure(figsize=[6,8])
for n in range(1,1+len(sys.argv[1:])) :
    # C_l
    plt.subplot(2,1,1)

    if os.path.isfile(sys.argv[n]+'_cl.txt'):
        cl=np.genfromtxt(sys.argv[n]+'_cl.txt').T
    elif  os.path.isfile(sys.argv[n]+'_cl.dat'):
        cl=np.genfromtxt(sys.argv[n]+'_cl.dat').T
    else:
        continue

    for i in range(1,1+len(cl[1:])) :
        plt.plot(cl[0],cl[i],label=sys.argv[n])

    plt.xlim(np.min(cl[0]),np.max(cl[0]))
    plt.ylabel(r"$C_\ell^{i,j}$")
    plt.xlabel(r"$\ell$")
    plt.legend()


    #c(theta)
    if os.path.isfile(sys.argv[n]+'_ctheta.txt'):
        ct=np.genfromtxt(sys.argv[n]+'_ctheta.txt').T
    else:
        continue

    plt.subplot(2,1,2)

    tdeg=np.degrees(ct[0])
    [plt.plot(tdeg,tdeg**2*ct[i],'+-',label=sys.argv[n]) for i in range(1,1+len(ct[1:]))]

    plt.xlim(0,10)
    plt.ylabel(r"$\theta^2 C_\theta^{i,j}$")
    plt.xlabel(r"$\theta$"+' [$^\circ$]')
    plt.legend()

plt.tight_layout()
plt.show(block=True)
