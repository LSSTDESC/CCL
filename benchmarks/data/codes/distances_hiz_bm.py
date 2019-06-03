import numpy as np
import matplotlib.pyplot as plt
import py_cosmo_mad as csm

TCMB=2.725
PLOT_STUFF=0
WRITE_STUFF=0
FS=16

def do_all(z_arr,cpar,prefix) :
    pcs=csm.PcsPar()
    pcs.background_set(cpar['om'],cpar['ol'],cpar['ob'],cpar['w0'],cpar['wa'],cpar['hh'],TCMB)

    a_arr=1./(z_arr+1)
    chi_arr=np.array([pcs.radial_comoving_distance(a) for a in a_arr])

    if PLOT_STUFF==1 :
        plt.plot(z_arr,chi_arr); plt.xlabel('$z$',fontsize=FS); plt.ylabel('$\\chi(z)\\,[{\\rm Mpc}\\,h^{-1}]$',fontsize=FS); plt.show()

    if WRITE_STUFF==1 :
        np.savetxt(prefix+"_chi.txt",np.transpose([z_arr,chi_arr]),header="[1] z, [2] chi(z) (Mpc/h)")

z_arr=np.array([10.,20.,50.,100.,200.,500.,1000.])

cpar_model1= {'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -1.0, 'wa': 0.0}
cpar_model2={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.0}
cpar_model3={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.1}

do_all(z_arr,cpar_model1,"model1_hiz")
do_all(z_arr,cpar_model2,"model2_hiz")
do_all(z_arr,cpar_model3,"model3_hiz")
