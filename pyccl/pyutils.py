
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


def _vectorize_fn_simple(fn, fn_vec, x, returns_status=True):    
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument (but no dependence on cosmology).
    """
    status = 0
    if isinstance(x, float):
        # Use single-value function
        if returns_status:
            f, status = fn(x, status)
        else:
            f = fn(x)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(x, x.size, status)
        else:
            f = fn_vec(x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(x, len(x), status)
        else:
            f = fn_vec(x, len(x))
    return f


def _vectorize_fn(fn, fn_vec, cosmo, x, returns_status=True):    
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    status = 0
    
    if isinstance(x, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x, status)
        else:
            f = fn(cosmo, x)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x, x.size, status)
        else:
            f = fn_vec(cosmo, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, x, len(x), status)
        else:
            f = fn_vec(cosmo, x, len(x))
    return f


def _vectorize_fn2(fn, fn_vec, cosmo, x, z, returns_status=True):
    """
    Generic wrapper to allow vectorized (1D array) access to CCL functions with 
    one vector argument and one scalar argument.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    status = 0
    
    if isinstance(x, float):
        # Use single-value function
        if returns_status:
            f, status = fn(cosmo, x, z, status) # Note order of x,z switched
        else:
            f = fn(cosmo, x, z)
    elif isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, z, x, x.size, status)
        else:
            f = fn_vec(cosmo, z, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, z, x, len(x), status)
        else:
            f = fn_vec(cosmo, z, x, len(x))
    return f
    
