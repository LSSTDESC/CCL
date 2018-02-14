from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn4

def p_1h(cosmo, k, a):
    """1halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        p_1h (float or array_like): 1 halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.p_1h, 
                          lib.p_1h, cosmo, k, a)
