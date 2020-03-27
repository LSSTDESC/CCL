import numpy as np
from scipy.interpolate import interp1d
from ..pyutils import _check_array_params
from ..background import growth_factor
from .. import ccllib as lib


def IA_norm(cosmo, z, a1=1.0, a1delta=None, a2=None, Om_m2_for_c2 = False, Om_m_fid=0.3):
    """
    Function to convert from a_ia values to c_ia values,
    using the standard convention of Blazek 2019 or the variant used
    by the Dark Energy Survey analysis.
    
    Args:
    cosmo (ccl cosmo object)
    z (float or array_like): z value(s) where amplitude is evaluated
    a1 (float or array_like): IA a1 at input z values. Defaults to 1.0
    a1delta (float or array_like): IA a1delta at input z values. Defaults to None
    a2 (float or array_like): IA a2 at input z values. Defaults to None
        
    Returns:
        c1 (float or array_like): IA c1 at input z values
        c1delta (float or array_like): IA c1delta at input z values
        c2 (float or array_like): IA c2 at input z values
    """
    gz = growth_factor(cosmo, 1./(1+z))
    c1=c2=c1delta=None
    a_arr = [a1,a2,a1delta]
    rho_crit = lib.cvar.constants.RHO_CRITICAL
    a_names = ['a1','a2','a1delta']
    for i,a in enumerate(a_arr):
        if np.ndim(a)>1:
            raise ValueError("%s should be a scalar or a 1-dim array" % a_names[i])
        if np.ndim(a)==1:
            try:
                len(a1)==len(z)
            except:
                raise ValueError("The array %s must have the same number of elements as z" % a_names[i])
    if a1 != None:
        c_1 = -1*a_1*5e-14*rho_crit*cosmo['Omega_m']/gz
    if a1delta != None:    
        c_1delta = -1*a_1delta*5e-14*rho_crit*cosmo['Omega_m']/gz
     if a2 != None:
        if Om_m2_for_c2:
            c_2 = a_2*5*5e-14*rho_crit*cosmo['Omega_m']**2/(Om_m_fid*gz**2) #Blazek2019 convention
        else:
            c_2 = a_2*5*5e-14*rho_crit*cosmo['Omega_m']/(gz**2) #DES convention
            
        
        
    return c1,c2,c1delta

class PTTracer(object):
    """PTTracers contain the information necessary to describe the
    perturbative, non-linear inhomogeneities associated with
    different physical quantities.

    In essence their main function is to store a set of redshift-
    dependent functions (e.g. perturbation theory biases) needed
    in a perturbation theory framework to provide N-point
    correlations.
    """
    def __init__(self):
        self.biases = {}
        self.type = None
        pass

    def get_bias(self, bias_name, z):
        """Get the value of one of the bias functions at a given
        redshift.

        Args:
            bias_name (str): name of the bias function to return.
            z (float or array_like): redshift.

        Returns:
            float or array_like: bias value at the input redshifts.
        """
        if bias_name not in self.biases:
            raise KeyError("Bias %s not included in this tracer" % bias_name)
        return self.biases[bias_name](z)

    def _get_bias_function(self, b):
        # If None, assume it's zero
        if b is None:
            b = 0

        # If it's a scalar, then assume it's a constant function
        if np.ndim(b) == 0:
            def _const(z):
                if np.ndim(z) == 0:
                    return b
                else:
                    return b * np.ones_like(z)

            return _const
        else:  # Otherwise interpolate
            z, b = _check_array_params(b)
            return interp1d(z, b, bounds_error=False,
                            fill_value=b[-1])


class PTMatterTracer(PTTracer):
    """:class:`PTTracer` representing matter fluctuations.
    """
    def __init__(self):
        self.biases = {}
        self.type = 'M'


class PTNumberCountsTracer(PTTracer):
    """:class:`PTTracer` representing number count fluctuations.
    This is described by 1st and 2nd-order biases and
    a tidal field bias. These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If `None`, a bias of zero is assumed.

    Args:
        b1 (float or tuple of arrays): a single number or a
            tuple of arrays (z, b(z)) giving the first-order
            bias.
        b2 (float or tuple of arrays): as above for the
            second-order bias.
        bs (float or tuple of arrays): as above for the
            tidal bias.
    """
    def __init__(self, b1, b2=None, bs=None):
        self.biases = {}
        self.type = 'NC'

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2
        self.biases['b2'] = self._get_bias_function(b2)
        # Initialize bs
        self.biases['bs'] = self._get_bias_function(bs)

    @property
    def b1(self):
        """Internal first-order bias function.
        """
        return self.biases['b1']

    @property
    def b2(self):
        """Internal second-order bias function.
        """
        return self.biases['b2']

    @property
    def bs(self):
        """Internal tidal bias function.
        """
        return self.biases['bs']


class PTIntrinsicAlignmentTracer(PTTracer):
    """:class:`PTTracer` representing intrinsic alignments.
    This is described by 1st and 2nd-order alignment biases
    and an overdensity bias. These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If `None`, a bias of zero is assumed.

    Args:
        c1 (float or tuple of arrays): a single number or a
            tuple of arrays (z, c1(z)) giving the first-order
            alignment bias.
        c2 (float or tuple of arrays): as above for the
            second-order alignment bias.
        cdelta (float or tuple of arrays): as above for the
            overdensity bias.
    """
    def __init__(self, c1, c2=None, cdelta=None):

        self.biases = {}
        self.type = 'IA'

        # Initialize c1
        self.biases['c1'] = self._get_bias_function(c1)
        # Initialize c2
        self.biases['c2'] = self._get_bias_function(c2)
        # Initialize cdelta
        self.biases['cdelta'] = self._get_bias_function(cdelta)

    @property
    def c1(self):
        """Internal first-order bias function.
        """
        return self.biases['c1']

    @property
    def c2(self):
        """Internal second-order bias function.
        """
        return self.biases['c2']

    @property
    def cdelta(self):
        """Internal overdensity bias function.
        """
        return self.biases['cdelta']
