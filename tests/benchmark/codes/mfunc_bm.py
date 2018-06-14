import numpy as np
import py_cosmo_mad as csm
import os

pcs=csm.PcsPar()
pcs.set_verbosity(1)
pcs.background_set(0.3,0.7,0.05,-1.,0.,0.7,2.7255)
pcs.set_linear_pk("BBKS",-3,3,0.01,0.96,0.8)
pcs.set_mf_params(10.,16.,0.01)

lm_arr=11.+4.*(np.arange(9)+0.0)/8.
m_arr=10.**lm_arr
z_arr=0.2*np.arange(7)
nm_arr=np.array([[pcs.multiplicity_function(m,z,"Tinker200")/pcs.dlsigMdlM(m) for m in m_arr] for z in z_arr])

head="[0]mass "
iz=1
for z in z_arr :
    head+="[%d]"%iz+"z=%g "%z
    iz+=1
np.savetxt("mfunc.txt",np.transpose(np.concatenate((m_arr[None,:],nm_arr),axis=0)),header=head)
