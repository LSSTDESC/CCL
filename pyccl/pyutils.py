
import ccllib as lib
import numpy as np
import core

def _cosmology_obj(cosmo):
    """
    Returns a ccl_cosmology object, given an input object which may be 
    ccl_cosmology, the Cosmology wrapper class, or an invalid type.
    """
    if isinstance(cosmo, lib.cosmology):
        return cosmo
    elif isinstance(cosmo, core.Cosmology):
        return cosmo.cosmo
    else:
        raise TypeError("Invalid Cosmology or ccl_cosmology object.")


def _vectorize_fn_simple(fn, fn_vec, x):    
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument (but no dependence on cosmology).
    """
    if isinstance(x, float):
        # Use single-value function
        return fn(x)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        return fn_vec(x, x.size)
    else:
        # Use vectorised function
        return fn_vec(x, len(x))


def _vectorize_fn(fn, fn_vec, cosmo, x):    
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    
    if isinstance(x, float):
        # Use single-value function
        return fn(cosmo, x)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        return fn_vec(cosmo, x, x.size)
    else:
        # Use vectorised function
        return fn_vec(cosmo, x, len(x))


def _vectorize_fn2(fn, fn_vec, cosmo, x, z):
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument and one scalar argument.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    
    if isinstance(x, float):
        # Use single-value function
        return fn(cosmo, x, z) # Note order of x,z switched
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        return fn_vec(cosmo, z, x, x.size)
    else:
        # Use vectorised function
        return fn_vec(cosmo, z, x, len(x))
        
