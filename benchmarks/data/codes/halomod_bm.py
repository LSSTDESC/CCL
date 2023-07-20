import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import os
from scipy.special import sici

cpar1={'Om_m':0.3,'Om_b':0.05,'Om_nu':0.,
       'h':0.70,'sig8':0.8,'n':0.96}
cpar2={'Om_m':0.272,'Om_b':0.0455,'Om_nu':0.,
       'h':0.704,'sig8':0.81,'n':0.967}
cpar3={'Om_m':0.3175,'Om_b':0.0490,'Om_nu':0.,
       'h':0.6711,'sig8':0.834,'n':0.9624}

USE_INT=False

LKHMIN=-4.
LKHMAX=1.
NKH=256

NM=2048
LM0=5.5
LMMIN=6.
LMMAX=16.

params={'DELTA_C':1.686470199841,
        'ST_A': 0.322,
        'ST_a': 0.707,
        'ST_p': 0.3,
        'DUF_A':7.85,
        'DUF_B':-0.081,
        'DUF_C':-0.71}

def st_gs(sigma,dc) :
    nu=dc/sigma
    anu2=params['ST_a']*nu*nu
    expnu=np.exp(-anu2*0.5)

    return params['ST_A']*np.sqrt(2*params['ST_a']/np.pi)*(1+(1./anu2)**params['ST_p'])*nu*expnu

def st_b1(sigma,dc) :
    nu=dc/sigma
    anu2=params['ST_a']*nu*nu

    return 1+(anu2-1+2*params['ST_p']/(1+anu2**params['ST_p']))/params['DELTA_C']

def duffy_c(m,z) :
    return params['DUF_A']*(m/2E12)**params['DUF_B']*(1+z)**params['DUF_C']

def get_halomod_pk(cp,karr,z,integrate=True,fname_out=None) :
    cosmo=ccl.Cosmology(Omega_c=cp['Om_m']-cp['Om_b'],Omega_b=cp['Om_b'],h=cp['h'],
                        sigma8=cp['sig8'],n_s=cp['n'],
                        transfer_function='eisenstein_hu',matter_power_spectrum='linear')

    lmarr=LMMIN+(LMMAX-LMMIN)*(np.arange(NM)+0.5)/NM #log(M*h/M_sum)
    dlm=(LMMAX-LMMIN)/NM
    lmarrb=np.zeros(NM+2); lmarrb[0]=lmarr[0]-dlm; lmarrb[1:-1]=lmarr; lmarrb[-1]=lmarr[-1]+dlm
    marr=10.**lmarr #M (in Msun/h) M_h Ms/h = M Ms
    marrb=10.**lmarrb #M (in Msun/h) M_h Ms/h = M Ms
    sigmarrb=ccl.sigmaM(cosmo,marrb/cp['h'],1./(1+z))
    sigmarr=sigmarrb[1:-1]
    dlsMdl10M=np.fabs((sigmarrb[2:]-sigmarrb[:-2])/(2*dlm))/sigmarr
    omega_z=cp['Om_m']*(1+z)**3/(cp['Om_m']*(1+z)**3+1-cp['Om_m'])
    delta_c=params['DELTA_C']*(1+0.012299*np.log10(omega_z))
    Delta_v=(18*np.pi**2+82*(omega_z-1)-39*(omega_z-1)**2)/omega_z
    fM=st_gs(sigmarr,delta_c)*dlsMdl10M
    bM=st_b1(sigmarr,delta_c)
    cM=duffy_c(marr,z)
    cMf=interp1d(lmarr,cM,kind='linear',bounds_error=False,fill_value=cM[0])
    rhoM=ccl.rho_x(cosmo,1.,'matter')/cp['h']**2

    #Compute mass function normalization
    fM0=1-np.sum(fM)*dlm
    fbM0=1-np.sum(fM*bM)*dlm

    rvM=(3*marr/(4*np.pi*rhoM*Delta_v))**0.333333333333
    rsM=rvM/cM
    rsM0=(3*10.**LM0/(4*np.pi*rhoM*Delta_v))**0.333333333/duffy_c(10.**LM0,z)
    rsMf=interp1d(lmarr,rsM,kind='linear',bounds_error=False,fill_value=rsM0)

    f1hf=interp1d(lmarr,marr*fM/rhoM,kind='linear',bounds_error=False,fill_value=0)
    f1h0=fM0*10.**LM0/rhoM
    f2hf=interp1d(lmarr,fM*bM,kind='linear',bounds_error=False,fill_value=0)
    f2h0=fbM0

    #NFW profile
    def u_nfw(lm,k) :
        x=k*rsMf(lm) #k*r_s
        c=cMf(lm)
        sic,cic=sici(x*(1+c))
        six,cix=sici(x)
        fsin=np.sin(x)*(sic-six)
        fcos=np.cos(x)*(cic-cix)
        fsic=np.sin(c*x)/(x*(1+c))
        fcon=np.log(1.+c)-c/(1+c)

        return (fsin+fcos-fsic)/fcon

    def p1h(k) :
        def integ(lm) :
            u=u_nfw(lm,k)
            f1h=f1hf(lm)
            return u*u*f1h
        if integrate :
            return quad(integ,lmarr[0],lmarr[-1])[0]+f1h0*(u_nfw(LM0,k))**2
        else :
            return np.sum(integ(lmarr))*dlm+f1h0*(u_nfw(LM0,k))**2
    
    def b2h(k) :
        def integ(lm) :
            u=u_nfw(lm,k)
            f2h=f2hf(lm)
            return u*f2h
        if integrate :
            return quad(integ,lmarr[0],lmarr[-1])[0]+f2h0*u_nfw(LM0,k)
        else :
            return np.sum(integ(lmarr))*dlm+f2h0*u_nfw(LM0,k)


    p1harr=np.array([p1h(kk) for kk in karr])
    b2harr=np.array([b2h(kk) for kk in karr])
    pklin=ccl.linear_matter_power(cosmo,karr*cp['h'],1./(1+z))*cp['h']**3

    p2harr=pklin*b2harr**2
    ptarr=p1harr+p2harr
    np.savetxt(fname_out,np.transpose([karr,pklin,p2harr,p1harr,ptarr]))
    return pklin,pklin*b2harr**2,p1harr,pklin*b2harr**2+p1harr

k_arr=10.**(LKHMIN+(LKHMAX-LKHMIN)*(np.arange(NKH)+0.5)/NKH)
for ip,cp in enumerate([cpar1,cpar2,cpar3]) :
    for z in [0,1] :
        get_halomod_pk(cp,k_arr,z+0.,integrate=USE_INT,fname_out='pk_hm_c%d_z%d.txt'%(ip+1,z))
