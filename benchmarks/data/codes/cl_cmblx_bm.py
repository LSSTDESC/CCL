import pyccl
import math
import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c,G
from astropy import units as u
from scipy.integrate import quad as scipy_int1d

c=c.to(u.km/u.second)

cosmo_fid=dict({'h':0.7,'s8':0.8,'Omc':0.3,'Og':0.0,'Omb':0.0,'Om':0.3,'m_nu':[0,0,0],
                'Neff':3.046,'Omk':0,'tau':0.06,'ns':0.96, 'Tcmb0':0})

# Reset astropy cosmology parameters to cosmo_fid. This is assuming flat LCDM and hence Omega_k=0
cosmo=cosmo.clone(H0=cosmo_fid['h']*100,Ob0=cosmo_fid['Omb'],Om0=cosmo_fid['Om'],
                  m_nu=cosmo_fid['m_nu']*u.eV,Neff=cosmo_fid['Neff'],Tcmb0=cosmo_fid['Tcmb0'])
cosmo_h=cosmo.clone(H0=100)

pk_params={'non_linear':0,'kmax':1E3,'kmin':5.e-5,'nk':5000}

class Power_Spectra():
    def __init__(self,cosmo_params=cosmo_fid,pk_params=pk_params,cosmo=cosmo,cosmo_h=None):
        self.cosmo_params=cosmo_params
        self.pk_params=pk_params
        self.cosmo=cosmo

        if not cosmo_h:
            self.cosmo_h=cosmo.clone(H0=100)
        else:
            self.cosmo_h=cosmo_h

    def Rho_crit(self,cosmo_h=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        rc=3*self.cosmo_h.H0**2/(8*np.pi*G2)
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc

    def DZ_int(self,z=[0],cosmo=None): #linear growth factor.. full integral.. eq 63 in Lahav and suto
        if not cosmo:
            cosmo=self.cosmo
        def intf(z):
            return (1+z)/(cosmo.H(z).value)**3
        j=0
        Dz=np.zeros_like(z,dtype='float32')

        for i in z:
            Dz[j]=cosmo.H(i).value*scipy_int1d(intf,i,np.inf,epsrel=1.e-6,epsabs=1e-6)[0]
            j=j+1
        Dz*=(2.5*cosmo.Om0*cosmo.H0.value**2)
        return Dz/Dz[0]

    def sigma_crit(self,zl=[],zs=[],cosmo_h=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        ds=cosmo_h.comoving_transverse_distance(zs)
        dl=cosmo_h.comoving_transverse_distance(zl)
        ddls=1-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        w=(3./2.)*((cosmo_h.H0/c)**2)*(1+zl)*dl/self.Rho_crit(cosmo_h)
        sigma_c=1./(ddls*w)
        x=ddls<=0 #zs<zl
        sigma_c[x]=np.inf
        return sigma_c

    def ccl_pk_bbks(self,z,cosmo_params=None,pk_params=None):
        if not cosmo_params:
            cosmo_params=self.cosmo_params
        if not pk_params:
            pk_params=self.pk_params

        cosmo_ccl=pyccl.Cosmology(h=cosmo_params['h'],Omega_c=cosmo_params['Omc'],Omega_b=cosmo_params['Omb'],
                                  sigma8=cosmo_params['s8'],n_s=cosmo_params['ns'],
                                  transfer_function='bbks', matter_power_spectrum='linear',
                                  Omega_g=cosmo_params['Og'])
                                  #m_nu=cosmo_params['m_nu'],Neff=cosmo_params['Neff'])
        kh=np.logspace(np.log10(pk_params['kmin']),np.log10(pk_params['kmax']),pk_params['nk'])
        nz=len(z)
        ps=np.zeros((nz,pk_params['nk']))
        ps0=[]
        z0=9.#PS(z0) will be rescaled using growth function when CCL fails.

        pyccl_pkf=pyccl.linear_matter_power
        if pk_params['non_linear']==1:
            pyccl_pkf=pyccl.nonlin_matter_power
        for i in np.arange(nz):
            try:
                ps[i]= pyccl_pkf(cosmo_ccl,kh,1./(1+z[i]))
            except Exception as err:
                print ('CCL err',err,z[i])
                if not np.any(ps0):
                    ps0=pyccl.linear_matter_power(cosmo_ccl,kh,1./(1.+z0))
                Dz=self.DZ_int(z=[z0,z[i]])
                ps[i]=ps0*(Dz[1]/Dz[0])**2
        return ps*cosmo_params['h']**3,kh/cosmo_params['h'] #factors of h to get in same units as camb output

    def cl_z(self,z=[],l=np.arange(3000)+1,pk_params=None,cosmo_h=None,
                cosmo=None,pk_func=None):
        if not cosmo_h:
            cosmo_h=self.cosmo_h
        if not pk_func:
            pk_func=self.camb_pk_too_many_z

        nz=len(z)
        nl=len(l)

        pk,kh=pk_func(z=z,pk_params=pk_params)

        cls=np.zeros((nz,nl),dtype='float32')*u.Mpc
        for i in np.arange(nz):
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value
            pk_int=interp1d(lz,pk[i]/DC_i**2,bounds_error=False,fill_value=0)
            cls[i][:]+=pk_int(l)*u.Mpc
        return cls

    def kappa_cl(self,zl_min=1.e-4,zl_max=1100,n_zl=10,log_zl=False,pk_func=None,
                zs1=[1100],p_zs1=[1],zs2=[1100],p_zs2=[1],
                pk_params=None,cosmo_h=None,l=np.arange(0,3001)):
        if not cosmo_h:
            cosmo_h=self.cosmo_h

        if log_zl:#bins for z_lens.
            zl=np.logspace(np.log10(max(zl_min,1.e-4)),np.log10(zl_max),n_zl)
        else:
            zl=np.linspace(zl_min,zl_max,n_zl)

        clz=self.cl_z(z=zl,l=l,cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func)
        # P(z,k=(l+05)/chi(z))/chi(z)^2 [nz,nl]
        clz=(clz.T*(c/(cosmo_h.efunc(zl)*cosmo_h.H0))).T
        # P(z,k=(l+0.5)/chi(z))/(chi(z)^2 H(z)) [nz,nl]
        #clz*=(c/(cosmo_h.efunc(zl)*cosmo_h.H0))

        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
        sigma_c1=rho/self.sigma_crit(zl=zl,zs=zs1,cosmo_h=cosmo_h) # 3/2 H0^2 (1+zl) chi(zl) * (1 - chi(zl)/chi(zs)) [n_zs,n_zl]
        sigma_c2=rho/self.sigma_crit(zl=zl,zs=zs2,cosmo_h=cosmo_h)

        dzl=np.gradient(zl)
        dzs1=np.gradient(zs1) if len(zs1)>1 else 1
        dzs2=np.gradient(zs2) if len(zs2)>1 else 1

        cl_zs_12=np.einsum('ji,ki,il',sigma_c2,sigma_c1*dzl,clz)#integrate over zl..
        cl=np.dot(p_zs2*dzs2,np.dot(p_zs1*dzs1,cl_zs_12))
        #Integral[ P(z,k=(l+0.5)/chi(z))/(H(z) chi(z)^2) Integral[3/2 H0^2 (1+z) p(zs1) chi(z) (1-chi(z)/chi(zs1)), dzs1] Integral[3/2 H0^2 (1+zl) chi(zl) (1+chi(zl)/chi(zs2)), dzs2], dz]
        cl/=np.sum(p_zs2*dzs2)*np.sum(p_zs1*dzs1)
        f=l*(l+1.)/(l+0.5)**2 #correction from Kilbinger+ 2017
        cl*=f**2
        #cl*=2./np.pi #comparison with CAMB requires this.
        return l,cl


    def g_kappa_cl(self,pk_func=None,zs=[1100],p_zs=[1],zg=[1],p_zg=[1],bias_g=1,
                pk_params=None,cosmo_h=None,l=np.arange(0,3001)):
        if not cosmo_h:
            cosmo_h=self.cosmo_h

        clz=self.cl_z(z=zg,l=l,cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func) # p(k)/chi**2
        clz*=bias_g

        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
        sigma_c=rho/self.sigma_crit(zl=zg,zs=zs,cosmo_h=cosmo_h)
        dzg=np.gradient(zg) if len(zg)>1 else 1
        dzs=np.gradient(zs) if len(zs)>1 else 1

        cl_zs_12=np.dot(sigma_c*dzg*p_zg,clz)#integrate over zl..
        cl=np.dot(p_zs*dzs,cl_zs_12)
        cl/=np.sum(p_zs*dzs)*np.sum(p_zg*dzg)
        return l,cl


if __name__ == "__main__":

    PS=Power_Spectra()
    #CMB lensing x galaxy clustering
    #Binned 1
    zg=np.genfromtxt('../codecomp_step2_outputs/bin1_histo.txt',
                 names=('z','pz'))
    l,cl_g_cmb=PS.g_kappa_cl(pk_func=PS.ccl_pk_bbks,zs=[1100],p_zs=[1],zg=zg['z'],p_zg=zg['pz'],bias_g=1,)
    np.savetxt("../codecomp_step2_outputs/run_b1b1histo_log_cl_dc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
    
    #Binned 2
    zg=np.genfromtxt('../codecomp_step2_outputs/bin2_histo.txt',
                 names=('z','pz'))
    l,cl_g_cmb=PS.g_kappa_cl(pk_func=PS.ccl_pk_bbks,zs=[1100],p_zs=[1],zg=zg['z'],p_zg=zg['pz'],bias_g=1,)
    np.savetxt("../codecomp_step2_outputs/run_b2b2histo_log_cl_dc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
    
    #Analytic
    sigz1=0.15
    zav=np.linspace(0,10.,10000)
    pza=1./(np.sqrt(2*math.pi)*sigz1)*np.exp(-0.5*((zav-1)/sigz1)**2)
    za={'z':zav,'pz':pza}
    l,cl_g_cmb=PS.g_kappa_cl(pk_func=PS.ccl_pk_bbks,zs=[1100],p_zs=[1],zg=za['z'],p_zg=za['pz'],bias_g=1,)
    np.savetxt("../codecomp_step2_outputs/run_b1b1analytic_log_cl_dc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
    
    #Analytic 2
    pza=1./(np.sqrt(2*math.pi)*sigz1)*np.exp(-0.5*((zav-1.5)/sigz1)**2)
    za={'z':zav,'pz':pza}
    l,cl_g_cmb=PS.g_kappa_cl(pk_func=PS.ccl_pk_bbks,zs=[1100],p_zs=[1],zg=za['z'],p_zg=za['pz'],bias_g=1,)
    np.savetxt("../codecomp_step2_outputs/run_b2b2analytic_log_cl_dc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
    
    #CMB lensing x galaxy lensing (kappa)
    #Binned 1
    zg=np.genfromtxt('../codecomp_step2_outputs/bin1_histo.txt',
                 names=('z','pz'))
    l,cl_g_cmb=PS.kappa_cl(pk_func=PS.ccl_pk_bbks,n_zl=1000,zs2=zg['z'],p_zs2=zg['pz'],zl_max=10)
    np.savetxt("../codecomp_step2_outputs/run_b1b1histo_log_cl_lc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])

    #Binned 2
    zg=np.genfromtxt('../codecomp_step2_outputs/bin2_histo.txt',
                     names=('z','pz'))
    l,cl_g_cmb=PS.kappa_cl(pk_func=PS.ccl_pk_bbks,n_zl=1000,zs2=zg['z'],p_zs2=zg['pz'],zl_max=10)
    np.savetxt("../codecomp_step2_outputs/run_b2b2histo_log_cl_lc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])

    #Analytic
    sigz1=0.15
    zav=np.linspace(0,10.,10000)
    pza=1./(np.sqrt(2*math.pi)*sigz1)*np.exp(-0.5*((zav-1)/sigz1)**2)
    za={'z':zav,'pz':pza}
    l,cl_g_cmb=PS.kappa_cl(pk_func=PS.ccl_pk_bbks,n_zl=1000,zs2=za['z'],p_zs2=za['pz'],zl_max=10)
    np.savetxt("../codecomp_step2_outputs/run_b1b1analytic_log_cl_lc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
    
    #Analytic 2
    pza=1./(np.sqrt(2*math.pi)*sigz1)*np.exp(-0.5*((zav-1.5)/sigz1)**2)
    za={'z':zav,'pz':pza}
    l,cl_g_cmb=PS.kappa_cl(pk_func=PS.ccl_pk_bbks,n_zl=1000,zs2=za['z'],p_zs2=za['pz'],zl_max=10)
    np.savetxt("../codecomp_step2_outputs/run_b2b2analytic_log_cl_lc.txt",np.column_stack((l,cl_g_cmb)),fmt=['%i','%.18e'])
