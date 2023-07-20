import numpy as np
import matplotlib.pyplot as plt
import py_cosmo_mad as csm

TCMB=2.725
PLOT_STUFF=0
WRITE_STUFF=1
FS=16
LKMAX=7

def do_all(z_arr,k_arr,cpar,prefix,wig):
    pcs=csm.PcsPar()
    pcs.background_set(cpar['om'],cpar['ol'],cpar['ob'],cpar['w0'],cpar['wa'],cpar['hh'],TCMB)
    if wig:
        typ = 'EH'
    else:
        typ = 'EH_smooth'
    pcs.set_linear_pk(typ,-3,LKMAX,0.01,cpar['ns'],cpar['s8'])

    gf0=pcs.growth_factor(1)
    a_arr=1./(z_arr+1)
    gf_arr=np.array([pcs.growth_factor(a) for a in a_arr])
    pk_arr=np.array([[pcs.Pk_linear_0(k)*(gf/gf0)**2 for k in k_arr] for gf in gf_arr])

    if PLOT_STUFF==1 :
        for i in np.arange(len(pk_arr)) :
            plt.plot(k_arr,pk_arr[i],label='$z=%.2lf$'%(z_arr[i]))
        plt.xlabel('$k\\,[h\\,{\\rm Mpc}^{-1}]$',fontsize=FS)
        plt.ylabel('$P(k)\\,[{\\rm Mpc}\\,h^{-1}]^3$',fontsize=FS)
        plt.gca().set_xscale('log');
        plt.gca().set_yscale('log');
        plt.legend(loc='lower left')
        plt.show()

    if WRITE_STUFF==1 :
        header_pk="[0] k (Mpc/h)^-1"
        for i in np.arange(len(z_arr)) :
            header_pk+=", [%d]"%(i+2)+" P(k,z=%.1lf) (Mpc/h)^3"%(z_arr[i])
        np.savetxt(prefix+"_pk_eh.txt",np.transpose(np.vstack((k_arr,pk_arr))),header=header_pk)

z_arr=np.array([0.,1.,2.,3.,4.,5.])
lk_arr=0.1*np.arange(41)-3.
k_arr=10**lk_arr

cpar_model1={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -1.0, 'wa': 0.0}

do_all(z_arr,k_arr,cpar_model1,"model1",True)
do_all(z_arr,k_arr,cpar_model1,"model1_nowig",False)
