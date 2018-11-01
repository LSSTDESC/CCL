from . import ccllib as lib
from .pyutils import _vectorize_fn2
from .core import check
import numpy as np

#TODO choices about interpolation/extrapolation

class Pk2D(object):
    """A power spectrum class holding the information needed to reconstruct an
    arbitrary function of wavenumber and scale factor.
    """
    def __init__(self,pkfunc=None,a_arr=None,lk_arr=None,pk_arr=None,is_logp=True) :
        """Constructor for Pk2D objects.

        Args:
            pkfunc (:obj:function): a function returning a floating point number or numpy array
                  with the signature `f(k,a)`, where k is a wavenumber (in units of Mpc^-1) and
                  a is the scale factor. The function must able to take numpy arrays as `k`. The
                  function must return the value(s) of the power spectrum (or its natural
                  logarithm, depending on the value of `is_logp`. The power spectrum units should
                  be compatible with those used by CCL (e.g. if you're passing a matter power
                  spectrum, its units should be Mpc^3).
            a_arr (array): an array holding values of the scale factor
            lk_arr (array): an array holding values of the natural logarithm of the wavenumber
                  (in units of Mpc^-1).
            pk_arr (array): a 2D array containing the values of the power spectrum at the values
                  of the scale factor and the wavenumber held by `a_arr` and `lk_arr`. The shape
                  of this array must be `[na,nk]`, where `na` is the size of `a_arr` and `nk` is
                  the size of `lk_arr`. This array can be provided in a flattened form as long as
                  the total size matches `nk*na`. The array can hold the values of the natural
                  logarithm of the power spectrum, depending on the value of `is_logp`. If `pkfunc`
                  is not None, then `a_arr`, `lk_arr` and `pk_arr` are ignored. However, either
                  `pkfunc` or all of the last three array must be non-None.
            is_logp (boolean): if True, pkfunc/pkarr return/hold the natural logarithm of the
                  power spectrum. Otherwise, the true value of the power spectrum is expected.
        """
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
        """Evaluate power spectrum.

        Args:
            k (float or array_like): wavenumber value(s) in units of Mpc^-1.
            a (float): value of the scale factor
            cosmo (:obj:`Cosmology`): Cosmology object. The cosmology object is needed in order
                  to evaluate the power spectrum outside the interpolation range in `a`. E.g.
                  if you want to evaluate the power spectrum at a very small a, not covered by
                  the arrays you passed when initializing this object, the power spectrum
                  will be extrapolated from the earliest available value using the linear growth
                  factor (for which a cosmology is needed).
            a_arr (array): an array holding values of the scale factor.

        Returns:
            float or array_like: value(s) of the power spectrum.
        """
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
        """Free memory associated with this Pk2D structure
        """
        if hasattr(self, 'has_psp'):
            if self.has_psp:
                lib.p2d_t_free(self.psp)
