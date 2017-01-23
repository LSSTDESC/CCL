
import ccllib as lib
from pyutils import _vectorize_fn, _vectorize_fn_simple

dNdz_types = {
    'nc':           lib.DNDZ_NC,
    'wl_cons':      lib.DNDZ_WL_CONS, 
    'wl_fid':       lib.DNDZ_WL_FID, 
    'wl_opt':       lib.DNDZ_WL_OPT
}

def bias_clustering(cosmo, a):
    return _vectorize_fn(lib.specs_bias_clustering, 
                         lib.specs_bias_clustering_vec, cosmo, a)

def sigmaz_clustering(z):
    return _vectorize_fn_simple(lib.specs_sigmaz_clustering, 
                                lib.specs_sigmaz_clustering_vec, z)

def sigmaz_sources(z):
    return _vectorize_fn_simple(lib.specs_sigmaz_sources, 
                                lib.specs_sigmaz_sources_vec, z)

def dNdz_tomog(z, dNdz_type, zmin, zmax, user_info):
    
    # FIXME: It's not clear how this interface should work.
    raise NotImplementedError()
    
    # Get dNdz type
    if dNdz_type not in dNdz_types.keys():
        raise ValueError("'%s' not a valid dNdz_type." % dNdz_type)
    
    tomoout = lib.specs_dNdz_tomog(z, dNdz_types[dNdz_type], zmin, zmax, user_info)
    print tomoout # FIXME: Is a status integer also being returned?
    return tomoout


# Provide aliases for functions to retain consistency with C API
specs_bias_clustering = bias_clustering
specs_sigmaz_clustering = sigmaz_clustering
specs_sigmaz_sources = sigmaz_sources
specs_dNdz_tomog = dNdz_tomog

