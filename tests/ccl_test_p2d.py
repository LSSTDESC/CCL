import numpy as np,math
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, decorators, run_module_suite
import pyccl as ccl
from pyccl import CCLError

def pk1d(k) :
    return (k/0.1)**(-1)

def grw(a) :
    return a

def pk2d(k,a) :
    return pk1d(k)*grw(a)


def reference_models():
    """
    Create a set of reference Cosmology() objects.
    """
    # Standard LCDM model
    cosmo1 = 1
    #ccl.Cosmology(
    #    Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

    return [cosmo1]

def all_finite(vals):
    """
    Returns True if all elements are finite (i.e. not NaN or inf).
    """
    return np.all( np.isfinite(vals) )

def check_p2d_init(cosmo):
    """
    Check that background and growth functions can be run.
    """

    ktest=1E-2; atest=0.5;
    
    #If no input
    assert_raises(TypeError, ccl.Pk2D)

    #Input function has incorrect signature
    assert_raises(TypeError, ccl.Pk2D, pkfunc=pk1d)
    ccl.Pk2D(pkfunc=pk2d)

    #Input arrays have incorrect sizes
    lkarr=-4.+6*np.arange(100)/99.
    aarr=0.05+0.95*np.arange(100)/99.
    pkarr=np.zeros([len(aarr),len(lkarr)])
    assert_raises(ValueError, ccl.Pk2D, a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr[1:])

    ccl.Pk2D(a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr)

def check_p2d_function(cosmo):
    psp=ccl.Pk2D(pkfunc=pk2d)
    assert_(np.fabs(psp.eval(ktest,atest)-pk2d)<1E-6)
    
def test_p2d_init():
    """
    Test background and growth functions in ccl.background.
    """
    for cosmo in reference_models():
        yield check_p2d_init, cosmo
    #yield check_p2d_function, 1

if __name__ == '__main__':
    run_module_suite()
