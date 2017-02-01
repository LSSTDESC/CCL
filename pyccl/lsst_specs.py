
import ccllib as lib
from pyutils import _vectorize_fn, _vectorize_fn_simple, np

dNdz_types = {
    'nc':           lib.DNDZ_NC,
    'wl_cons':      lib.DNDZ_WL_CONS, 
    'wl_fid':       lib.DNDZ_WL_FID, 
    'wl_opt':       lib.DNDZ_WL_OPT
}

class PhotoZFunction(object):
    
    def __init__(self, func, args=None):
        """
        Create a new photo-z function.
        
        Parameters
        ----------
        func : Callable function object
            Must have the call signature func(z_ph, z_s, args).
        
        args : tuple
            Tuple of arguments to be passed as the third argument of func().
        """
        # Wrap user-defined function up so that only two args are needed 
        # at run-time
        _func = lambda z_ph, z_s: func(z_ph, z_s, args)
        
        # Create user_pz_info object
        self.pz_func = lib.specs_create_photoz_info_from_py(_func)
    
    def __del__(self):
        """
        Destructor for PhotoZFunction object.
        """
        try:
            lib.specs_free_photoz_info(self.pz_func)
        except:
            pass


def bias_clustering(cosmo, a):
    return _vectorize_fn(lib.specs_bias_clustering, 
                         lib.specs_bias_clustering_vec, cosmo, a)

def sigmaz_clustering(z):
    return _vectorize_fn_simple(lib.specs_sigmaz_clustering, 
                                lib.specs_sigmaz_clustering_vec, z)

def sigmaz_sources(z):
    return _vectorize_fn_simple(lib.specs_sigmaz_sources, 
                                lib.specs_sigmaz_sources_vec, z)


def dNdz_tomog(z, dNdz_type, zmin, zmax, pz_func):
    
    # Ensure that an array will be passed to specs_dNdz_tomog_vec
    z = np.atleast_1d(z)
    
    # Do type-check for pz_func argument
    if not isinstance(pz_func, PhotoZFunction):
        raise TypeError("pz_func must be a PhotoZFunction instance.")
    
    # Get dNdz type
    if dNdz_type not in dNdz_types.keys():
        raise ValueError("'%s' not a valid dNdz_type." % dNdz_type)
    
    # Call dNdz tomography function
    dNdz = lib.specs_dNdz_tomog_vec( dNdz_types[dNdz_type], zmin, zmax, 
                                     pz_func.pz_func, z, z.size )
    return dNdz

# Provide aliases for functions to retain consistency with C API
specs_bias_clustering = bias_clustering
specs_sigmaz_clustering = sigmaz_clustering
specs_sigmaz_sources = sigmaz_sources
specs_dNdz_tomog = dNdz_tomog

