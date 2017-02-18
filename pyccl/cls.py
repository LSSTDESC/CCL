
import ccllib as lib
import constants as const
from pyutils import _cosmology_obj, check
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
    """ClTracer class used to wrap the cl_tracer found
    in CCL.

    A ClTracer is a data structure that contains all information
    describing the transfer functon of one tracer of the matter
    distribution.

    """

    def __init__(self, cosmo, tracer_type=None, has_rsd=False, 
                 has_magnification=False, has_intrinsic_alignment=False, 
                 z_n=None, n=None, z_b=None, b=None, 
                 z_s=None, s=None, z_ba=None, ba=None, 
                 z_rf=None, rf=None):
        """Creates the ClTracer.

        Note: unless otherwise stated defaults are None.

        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            tracer_type (:obj:`str`): Specifies what tracer to use.
            has_rsd (bool, optional): Flag to incorporate RSD into the model for CL. Defaults to False.
            has_magnification (bool, optional): Flag to incorporate magnification into the model for Cl. Defaults to False.
            has_intrinsic_alignment (bool, optional): Flag to incorporate intrinsic alignment into the model. Defaults to False.
            z_n (array_like, optional): Array of redshifts for N(z).
            n (array_like, optional): Array of N(z)-values.
            z_b (array_like, optional): Array of redshifts for alignment biases, b(z).
            b (array_like, optional): Array of alignment biases.
            z_s (array_like, optional): Array of redshifts for shapes, s(z).
            s (array_like, optional): Array of shapes.
            z_ba (array_like, optional): Array of redshifts for intrinsic alignment amplitudes.
            ba (array_like, optional): Array of intrinsic alignment amplitudes.
            z_rf (array_like, optional): Array of redshifts for the red fraction, rf(z).
            rf (array_like, optional): Array of red fractions.

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
        status = 0
        self.cltracer, status = lib.cl_tracer_new_wrapper(
                            cosmo, 
                            tracer_types[tracer_type],
                            int(has_rsd), 
                            int(has_magnification), 
                            int(has_intrinsic_alignment),
                            z_n, n, z_b, b, z_s, s, z_ba, ba, z_rf, rf, status )
        # TODO: worry about the status
        
    def __del__(self):
        """Free memory associated with CCL_ClTracer object.

        """
        lib.cl_tracer_free(self.cltracer)


class ClTracerNumberCounts(ClTracer):
    """ClTracer for galaxy clustering, N(z).

    """
    
    def __init__(self, cosmo, has_rsd, has_magnification, 
                 z_n, n, z_b, b, z_s=None, s=None):
        """ClTracer for galaxy clustering, N(z).

        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            has_rsd (bool, optional): Flag to incorporate RSD into the model for CL. Defaults to False.
            has_magnification (bool, optional): Flag to incorporate magnification into the model for Cl. Defaults to False.
            z_n (array_like): Array of redshifts for N(z).
            n (array_like): Array of N(z)-values.
            z_b (array_like): Array of redshifts for biases, b(z).
            b (array_like): Array of biases.
            z_s (array_like, optional): Array of redshifts for shapes, s(z). Not optional if has_magnification is True.
            s (array_like, optional): Array of shapes. Not optional if has_magnification is True.

        """
        
        # Sanity check on input arguments
        if has_magnification and (z_s is None or s is None):
                raise ValueError("Keyword args (z_s, s) must be specified if "
                                 "has_magnification=True.")
        
        # Call ClTracer constructor with appropriate arguments
        super(ClTracerNumberCounts, self).__init__(
                 cosmo=cosmo, tracer_type='nc', 
                 has_rsd=has_rsd, has_magnification=has_magnification, 
                 has_intrinsic_alignment=False, 
                 z_n=z_n, n=n, z_b=z_b, b=b, z_s=z_s, s=s, 
                 z_ba=None, ba=None, z_rf=None, rf=None)


class ClTracerLensing(ClTracer):
    """ClTracer for lensing shapes, s(z).

    """
    
    def __init__(self, cosmo, has_intrinsic_alignment, 
                 z_n, n, z_ba=None, ba=None, z_rf=None, rf=None):
        """ClTracer for lensing shapes, s(z).
        
        Args:
            cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object.
            has_intrinsic_alignment (bool, optional): Flag to incorporate intrinsic alignment into the model. Defaults to False.
            z_n (array_like): Array of redshifts for N(z).
            n (array_like): Array of N(z)-values.
            z_ba (array_like, optional): Array of redshifts for biases, b(z). Not optional if has_intrinsic_alignment is True.
            ba (array_like, optional): Array of biases. Not optional if has_intrinsic_alignment is True.
            z_rf (array_like, optional): Array of redshifts for the red fraction, rf(z). Not optional if has_intrinsic_alignment is True.
            rf (array_like, optional): Array of red fractions. Not optional if has_intrinsic_alignment is True.

        """
        
        # Sanity check on input arguments
        if has_intrinsic_alignment \
        and (z_ba is None or ba is None or z_rf is None or rf is None):
                raise ValueError("Keyword args (z_ba, ba, z_rf, rf) must be "
                                 "specified if has_intrinsic_alignment=True.")
        
        # Call ClTracer constructor with appropriate arguments
        super(ClTracerLensing, self).__init__(
                 cosmo=cosmo, tracer_type='wl', 
                 has_rsd=False, has_magnification=False, 
                 has_intrinsic_alignment=has_intrinsic_alignment, 
                 z_n=z_n, n=n, z_b=None, b=None, z_s=None, s=None, 
                 z_ba=z_ba, ba=ba, z_rf=z_rf, rf=rf)


def _cltracer_obj(cltracer):
    """Returns a CCL_ClTracer object, given an input object which may be 
    CCL_ClTracer, the ClTracer wrapper class, or an invalid type.

    Args:
        cltracer (:obj:): Either a CCL_ClTracer or the ClTracer wrapper class.

    Returns:
        cltracer (:obj:): Either a CCL_ClTracer or the ClTracer wrapper class.

    """
    # TODO: Is ClTracer a valid type?
    if isinstance(cltracer, lib.CCL_ClTracer):
        return cltracer
    elif isinstance(cltracer, ClTracer):
        return cltracer.cltracer
    else:
        raise TypeError("Invalid ClTracer or CCL_ClTracer object.")


def angular_cl(cosmo, cltracer1, cltracer2, ell):
    """Calculate angular power spectrum for two tracers.

    Args:
        cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object
        cltracer1 (:obj:): A Cl tracer.
        cltracer2 (:obj:): A Cl tracer.
        ell (float or array_like): Angular wavenumber to evaluate the tracers at.

    Returns:
        cl (float or array_like): Cl values for the tracers at `ell`.

    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    
    # Access CCL_ClTracer objects
    clt1 = _cltracer_obj(cltracer1)
    clt2 = _cltracer_obj(cltracer2)
    
    status = 0
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int) :
        # Use single-value function
        cl, status = lib.angular_cl(cosmo, ell, clt1, clt2, status)
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        cl, status = lib.angular_cl_vec(cosmo, clt1, clt2, ell, ell.size, status)
    else:
        # Use vectorised function
        cl, status = lib.angular_cl_vec(cosmo, clt1, clt2, ell, len(ell), status)
    check(status)
    return cl
