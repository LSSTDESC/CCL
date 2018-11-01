from . import ccllib as lib
from .pyutils import _vectorize_fn2
from .core import check
import numpy as np

#TODO choices about interpolation/extrapolation

class Pk2D(object):
    def __init__(self,pkfunc=None,a_arr=None,lk_arr=None,pk_arr=None,is_logp=True) :
        status=0
        if(pkfunc is None) : #Initialize power spectrum from 2D array
            #Make sure input makes sense
            if (a_arr is None) or (lk_arr is None) or (pk_arr is None) :
                raise TypeError("If you do not provide a function, you must provide arrays")

            pkflat=pk_arr.flatten()
            #Check dimensions make sense
            if (len(a_arr)*len(lk_arr) != len(pkflat)) :
                raise ValueError("Size of input arrays is inconsistent")
        else : #Initialize power spectrum from function
            #Check that the input function has the right signature
            try :
                f=pkfunc(k=np.array([1E-2,2E-2]),a=0.5)
            except :
                raise TypeError("Can't use input function")

            #Set k and a sampling from CCL parameters
            nk=lib.get_pk_spline_nk()
            na=lib.get_pk_spline_na()
            a_arr,status=lib.get_pk_spline_a(na,status)
            check(status)
            lk_arr,status=lib.get_pk_spline_lk(nk,status)
            check(status)

            #Compute power spectrum on 2D grid
            pkflat=np.zeros([na,nk])
            for ia,a in enumerate(a_arr) :
                pkflat[ia,:]=pkfunc(k=np.exp(lk_arr),a=a)
            pkflat=pkflat.flatten()
            
        self.psp,status=lib.set_p2d_new_from_arrays(lk_arr,a_arr,pkflat,int(is_logp),status)
        check(status)
        self.has_psp=True

    def eval(self,k,a,cosmo=None) :
        status=0
        if cosmo is not None :
            cospass=cosmo.cosmo
        else :
            raise NotImplementedError("Currently we need a cosmology to extrapolate growth")
            cospass=None
            
        if isinstance(k,int) :
            k=float(k)
        if isinstance(k,float) :
            f,status=lib.p2d_eval_single(self.psp,np.log(k),a,cospass,status)
        elif isinstance(k,np.ndarray) :
            f,status=lib.p2d_eval_multi(self.psp,np.log(k),a,cospass,k.size,status)
        else :
            f,status=lib.p2d_eval_multi(self.psp,np.log(k),a,cospass,len(k),status)
        check(status,cosmo)

        return f
        raise NotImplementedError("Not implemented yet")
    
    def __del__(self) :
        if hasattr(self, 'has_psp'):
            if self.has_psp:
                lib.p2d_t_free(self.psp)
        #raise NotImplementedError("Not implemented yet")
