import numpy as np
import matplotlib.pyplot as plt
import py_cosmo_mad as csm

#Produces all sigma(M) benchmarks
#Contact david.alonso@physics.ox.ac.uk if you have issues running this script

TCMB=2.725
PLOT_STUFF=0
WRITE_STUFF=1
FS=16
LKMAX=7

def do_all(m_arr,cpar,prefix) :
    pcs=csm.PcsPar()
    pcs.background_set(cpar['om'],cpar['ol'],cpar['ob'],cpar['w0'],cpar['wa'],cpar['hh'],TCMB)
    pcs.set_linear_pk('BBKS',-3,LKMAX,0.01,cpar['ns'],cpar['s8'])

    r_arr=np.array([pcs.M2R(m) for m in m_arr])
    sm_arr=np.sqrt(np.array([pcs.sig0_L(r,r,'TopHat','TopHat') for r in r_arr]))

    if PLOT_STUFF==1 :
        plt.plot(m_arr,sm_arr)
        plt.xlabel('$M\\,[M_{\\odot}\\,h^{-1}]$',fontsize=FS)
        plt.ylabel('$\\sigma(M)$',fontsize=FS)
        plt.gca().set_xscale('log');
        plt.gca().set_yscale('log');
        plt.show()

    if WRITE_STUFF==1 :
        np.savetxt(prefix+"_sm.txt",np.transpose([m_arr,sm_arr]),header="[1] M (M_sun/h), [2] sigma(M)")

z_arr=np.array([0.,1.,2.,3.,4.,5.])
lm_arr=6.+2*np.arange(6)
m_arr=10**lm_arr

cpar_model1={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -1.0, 'wa': 0.0}
cpar_model2={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.0}
cpar_model3={'om': 0.3,'ol': 0.7,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.1}

do_all(m_arr,cpar_model1,"model1")
do_all(m_arr,cpar_model2,"model2")
do_all(m_arr,cpar_model3,"model3")
