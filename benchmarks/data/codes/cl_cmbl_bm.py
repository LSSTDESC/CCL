import pyccl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import astropy.cosmology as csmm
from astropy.constants import c,G
from astropy import units as u
#If you have issues running this script, contact Sukhdeep Singh (nabsukh@gmail.com)

cosmo=csmm.FlatLambdaCDM(70.,0.3,Tcmb0=2.725)
cosmo_h=cosmo.clone(H0=100)
c=c.to(u.km/u.second)

cosmo_fid=dict({'h':0.7,'Omb':0.0,'Omd':0.3,'s8':0.8,'Om':0.3,
                'As':2.12e-09,'mnu':0.,'Omk':0.,'tau':0.06,'ns':0.96})
pk_params={'non_linear':1,'kmax':30,'kmin':1.e-4,'nk':5000}

class Power_Spectra():
    def __init__(self,cosmo_params=cosmo_fid,pk_params=pk_params,cosmo=cosmo,cosmo_h=None):
        self.cosmo_params=cosmo_params
        self.pk_params=pk_params
        self.cosmo=cosmo

        if not cosmo_h:
            self.cosmo_h=cosmo.clone(H0=100)
        else:
            self.cosmo_h=cosmo_h
            
    def ccl_pk(self,z,cosmo_params=None,pk_params=None):
        if not cosmo_params:
            cosmo_params=self.cosmo_params
        if not pk_params:
            pk_params=self.pk_params

        cosmo_ccl=pyccl.Cosmology(h=cosmo_params['h'],Omega_c=cosmo_params['Omd'],
                                  Omega_b=cosmo_params['Omb'],sigma8=0.8,
                                  n_s=cosmo_params['ns'],m_nu=cosmo_params['mnu'],
                                  transfer_function='bbks')
        kh=np.logspace(np.log10(pk_params['kmin']),np.log10(pk_params['kmax']),pk_params['nk'])
        nz=len(z)
        ps=np.zeros((nz,pk_params['nk']))
        ps0=[]
        z0=9.#PS(z0) will be rescaled using growth function when CCL fails. 

        pyccl_pkf=pyccl.linear_matter_power
        if pk_params['non_linear']==1:
            pyccl_pkf=pyccl.nonlin_matter_power
        for i in np.arange(nz):
            ps[i]= pyccl_pkf(cosmo_ccl,kh,1./(1+z[i]))
        return ps*cosmo_params['h']**3,kh/cosmo_params['h'] #factors of h to get in same units as camb output

    def cl_z(self,z=[],l=np.arange(3000)+1,pk_params=None,cosmo_h=None,cosmo=None,pk_func=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h

        nz=len(z)
        nl=len(l)

        pk,kh=pk_func(z=z,pk_params=pk_params)

        cls=np.zeros((nz,nl),dtype='float32')*u.Mpc**2
        for i in np.arange(nz):
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value
            pk_int=interp1d(lz,pk[i]/DC_i**2,bounds_error=False,fill_value=0)
            cls[i][:]+=pk_int(l)*u.Mpc*(c/(cosmo_h.efunc(z[i])*cosmo_h.H0))
        return cls

    def kappa_cl(self,zl_min=0,zl_max=1100,n_zl=10,log_zl=False,pk_func=None,
                zs1=[1100],p_zs1=[1],zs2=[1100],p_zs2=[1],
                pk_params=None,cosmo_h=None,l=np.arange(3001)):
        if not cosmo_h:
            cosmo_h=self.cosmo_h

        if log_zl:#bins for z_lens. 
            zl=np.logspace(np.log10(max(zl_min,1.e-4)),np.log10(zl_max),n_zl)
        else:
            zl=np.linspace(zl_min,zl_max,n_zl)

        ds1=cosmo_h.comoving_transverse_distance(zs1)
        ds2=cosmo_h.comoving_transverse_distance(zs2) 
        dl=cosmo_h.comoving_transverse_distance(zl)
        prefac=1.5*(cosmo_h.H0/c)**2*cosmo_h.Om0
        sigma_c1=prefac*(1+zl)*dl*(1-np.multiply.outer(1./ds1,dl))
        sigma_c2=prefac*(1+zl)*dl*(1-np.multiply.outer(1./ds2,dl))
        clz=self.cl_z(z=zl,l=l,cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func)

        dzl=np.gradient(zl)
        dzs1=np.gradient(zs1) if len(zs1)>1 else 1
        dzs2=np.gradient(zs2) if len(zs2)>1 else 1

        cl_zs_12=np.einsum('ji,ki,il',sigma_c2,sigma_c1*dzl,clz)#integrate over zl..
        cl=np.dot(p_zs2*dzs2,np.dot(p_zs1*dzs1,cl_zs_12))
        cl/=np.sum(p_zs2*dzs2)*np.sum(p_zs1*dzs1)
        f=l*(l+1.)/(l+0.5)**2 #correction from Kilbinger+ 2017
        cl*=f**2
        return l,cl


if __name__ == "__main__":
    PS=Power_Spectra()
    l,cl2=PS.kappa_cl(n_zl=10000,log_zl=True,zl_min=1.e-4,zl_max=1100,pk_func=PS.ccl_pk)
    
    plt.figure()
    plt.plot(l,cl2)
    plt.loglog()
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C^{\\kappa\\kappa}_\\ell$',fontsize=16)
    plt.show()
    
    np.savetxt('benchmark_cc.txt',np.transpose([l,cl2]),fmt='%d %lE')
