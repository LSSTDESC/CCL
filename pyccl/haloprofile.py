from . import ccllib as lib
from .pyutils import _vectorize_fn2, _vectorize_fn4


#ccl_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status)
#cosmo dependence + vectorized argument + 4 float arguments. Do I need a _vectorize_fn6?
