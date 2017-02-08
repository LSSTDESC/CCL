
import ccllib as lib
import constants as const
from pyutils import _cosmology_obj
import numpy as np

# Mapping between names for tracers and internal CCL tracer types
tracer_types = {
    'nc':               const.CL_TRACER_NC,
    'number_count':     const.CL_TRACER_NC,
    'wl':               const.CL_TRACER_WL,
    'lensing':          const.CL_TRACER_WL,
    'weak_lensing':     const.CL_TRACER_WL,
}

# Define symbolic 'None' type for arrays, to allow proper handling by swig wrapper
NoneArr = np.array([])

class ClTracer(object):
    
    def __init__(self, cosmo, tracer_type=None, has_rsd=False, 
                 has_magnification=False, has_intrinsic_alignment=False, 
                 z_n=None, n=None, z_b=None, b=None, 
                 z_s=None, s=None, z_ba=None, ba=None, 
                 z_rf=None, rf=None):
        """
        Object handling a ClTracer (tracer with an angular power spectrum).
        """
        # Verify cosmo object
        cosmo = _cosmology_obj(cosmo)
        
        # Check tracer type
        if tracer_type not in tracer_types.keys():
            raise KeyError("'%s' is not a valid tracer_type." % tracer_type)
        
        # Convert array arguments that are 'None' into 'NoneArr' type
        if n is None: n = NoneArr
        if b is None: b = NoneArr
        if s is None: s = NoneArr
        if ba is None: ba = NoneArr
        if rf is None: rf = NoneArr
        if z_n is None: z_n = NoneArr
        if z_b is None: z_b = NoneArr
        if z_s is None: z_s = NoneArr
        if z_ba is None: z_ba = NoneArr
        if z_rf is None: z_rf = NoneArr
        
        # Construct new ccl_cl_tracer
        self.cltracer = lib.cl_tracer_new_wrapper(
                            cosmo, 
                            tracer_types[tracer_type],
                            int(has_rsd), 
                            int(has_magnification), 
                            int(has_intrinsic_alignment),
                            z_n, n, z_b, b, z_s, s, z_ba, ba, z_rf, rf )
        
    def __del__(self):
        """
        Free memory associated with CCL_ClTracer object.
        """
        lib.cl_tracer_free(self.cltracer)


class ClTracerNumberCounts(ClTracer):
    
    def __init__(self, cosmo, has_rsd, has_magnification, 
                 z_n, n, z_b, b, z_s=None, s=None):
        
        # Call ClTracer constructor with appropriate arguments
        super(ClTracerNumberCounts, self).__init__(
                 cosmo=cosmo, tracer_type='nc', 
                 has_rsd=has_rsd, has_magnification=has_magnification, 
                 has_intrinsic_alignment=False, 
                 z_n=z_n, n=n, z_b=z_b, b=b, z_s=z_s, s=s, 
                 z_ba=None, ba=None, z_rf=None, rf=None)


class ClTracerLensing(ClTracer):
    
    def __init__(self, cosmo, has_intrinsic_alignment, 
                 z_n, n, z_ba=None, ba=None, z_rf=None, rf=None):
        
        # Call ClTracer constructor with appropriate arguments
        super(ClTracerLensing, self).__init__(
                 cosmo=cosmo, tracer_type='wl', 
                 has_rsd=False, has_magnification=False, 
                 has_intrinsic_alignment=has_intrinsic_alignment, 
                 z_n=z_n, n=n, z_b=None, b=None, z_s=None, s=None, 
                 z_ba=z_ba, ba=ba, z_rf=z_rf, rf=rf)


def _cltracer_obj(cltracer):
    """
    Returns a CCL_ClTracer object, given an input object which may be 
    CCL_ClTracer, the ClTracer wrapper class, or an invalid type.
    """
    # FIXME: Is ClTracer a valid type?
    if isinstance(cltracer, lib.CCL_ClTracer):
        return cltracer
    elif isinstance(cltracer, ClTracer):
        return cltracer.cltracer
    else:
        raise TypeError("Invalid ClTracer or CCL_ClTracer object.")


def angular_cl(cosmo, cltracer1, cltracer2, ell):
    """
    Calculate angular power spectrum for two tracers.
    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    
    # Access CCL_ClTracer objects
    clt1 = _cltracer_obj(cltracer1)
    clt2 = _cltracer_obj(cltracer2)
    
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int) :
        # Use single-value function
        return lib.angular_cl(cosmo, ell, clt1, clt2)
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        return lib.angular_cl_vec(cosmo, clt1, clt2, ell, ell.size)
    else:
        # Use vectorised function
        return lib.angular_cl_vec(cosmo, clt1, clt2, ell, len(ell))

