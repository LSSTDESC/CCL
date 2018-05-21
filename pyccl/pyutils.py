
from pyccl import ccllib as lib
import numpy as np
import pyccl.core

def check(status, cosmo=None):
    """Check the status returned by a ccllib function.

    Args:
        status (int or core.error_types): Flag or error describing the success of a function.

    """
    # Check for normal status (no action required)
    if status == 0: return
    
    # Get status message from Cosmology object, if there is one
    if cosmo is not None:
        msg = _cosmology_obj(cosmo).status_message
    else:
        msg = ""

    # Check for known error status
    if status in pyccl.core.error_types.keys():
        raise RuntimeError("Error %s: %s" % (pyccl.core.error_types[status], msg))

    # Check for unknown error
    if status != 0:
        raise RuntimeError("Error %d: %s" % (status, msg))


def debug_mode(debug):
    """Toggle debug mode on or off. If debug mode is on, the C backend is 
    forced to print error messages as soon as they are raised, even if the 
    flow of the program continues. This makes it easier to track down errors. 
    
    If debug mode is off, the C code will not print errors, and the Python 
    wrapper will raise the last error that was detected. If multiple errors 
    were raised, all but the last will be overwritten within the C code, so the 
    user will not necessarily be informed of the root cause of the error.

    Args:
        debug (bool): Switch debug mode on (True) or off (False).

    """
    if debug:
        lib.set_debug_policy(lib.CCL_DEBUG_MODE_ON)
    else:
        lib.set_debug_policy(lib.CCL_DEBUG_MODE_OFF)
    

def _cosmology_obj(cosmo):
    """Returns a ccl_cosmology object, given an input object which may be
    ccl_cosmology, the Cosmology wrapper class, or an invalid type.

    Args:
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets converted to a ccl_cosmology.

    """
    if isinstance(cosmo, lib.cosmology):
        return cosmo
    elif isinstance(cosmo, pyccl.core.Cosmology):
        return cosmo.cosmo
    else:
        raise TypeError("Invalid Cosmology or ccl_cosmology object.")


def _vectorize_fn_simple(fn, fn_vec, x, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument (but no dependence on cosmology).

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in a .i file.
        x (float or array_like): Argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    status = 0
    if isinstance(x, int): x = float(x)
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

    # Check result and return
    check(status)
    return f

def _vectorize_fn(fn, fn_vec, cosmo, x, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)
    
    status = 0
    
    if isinstance(x, int): x = float(x)
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

    # Check result and return
    check(status, cosmo_in)
    return f


def _vectorize_fn2(fn, fn_vec, cosmo, x, z, returns_status=True):

    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument and one scalar argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
        z (float): Scalar argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)
    status = 0
    scalar = False

    # If a scalar was passed, convert to an array
    if isinstance(x, int): x = float(x)
    if isinstance(x, float):
        scalar = True
        x = np.array([x,])

    if isinstance(x, np.ndarray):
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

    # Check result and return
    check(status, cosmo_in)
    if scalar:
        return f[0]
    else:
        return f

def _vectorize_fn3(fn, fn_vec, cosmo, x, n, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument and one integer argument, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
        n (int): Integer argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)
    status = 0
    scalar = False
    
    if isinstance(x, int): x = float(x)
    if isinstance(x, float):
        scalar = True
        x=np.array([x,])

    if isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, n, x, x.size, status)
        else:
            f = fn_vec(cosmo, n, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, n, x, len(x), status)
        else:
            f = fn_vec(cosmo, n, x, len(x))

    # Check result and return
    check(status, cosmo_in)
    if scalar:
        return f[0]
    else:
        return f

def _vectorize_fn4(fn, fn_vec, cosmo, x, a, d, returns_status=True):
    """Generic wrapper to allow vectorized (1D array) access to CCL functions with
    one vector argument and two float arguments, with a cosmology dependence.

    Args:
        fn (callable): Function with a single argument.
        fn_vec (callable): Function that has a vectorized implementation in a .i file.
        cosmo (ccl_cosmology or Cosmology): The input cosmology which gets converted to a ccl_cosmology.
        x (float or array_like): Argument to fn.
        a (float): Float argument to fn.
        d (float): Float argument to fn.
        returns_stats (bool): Indicates whether fn returns a status.

    """
    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)
    status = 0
    scalar = False
    
    if isinstance(x, int): x = float(x)
    if isinstance(x, float):
        scalar = True
        x=np.array([x,])

    if isinstance(x, np.ndarray):
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, a, d, x, x.size, status)
        else:
            f = fn_vec(cosmo, a, d, x, x.size)
    else:
        # Use vectorised function
        if returns_status:
            f, status = fn_vec(cosmo, a, d, x, len(x), status)
        else:
            f = fn_vec(cosmo, a, d, x, len(x))

    # Check result and return
    check(status, cosmo_in)
    if scalar:
        return f[0]
    else:
        return f

