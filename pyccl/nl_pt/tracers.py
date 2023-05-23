__all__ = ("translate_IA_norm", "PTTracer", "PTMatterTracer",
           "PTNumberCountsTracer", "PTIntrinsicAlignmentTracer",)

from abc import abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from .. import CCLAutoRepr, physical_constants
from ..pyutils import _check_array_params


def translate_IA_norm(cosmo, *, z, a1=1.0, a1delta=None, a2=None,
                      Om_m2_for_c2=False, Om_m_fid=0.3):
    """
    Function to convert from :math:`A_{ia}` values to :math:`c_{ia}` values,
    for the intrinsic alignment bias parameters using the standard
    convention of `Blazek et al. 2019 <https://arxiv.org/abs/1708.09247>`_
    or the variant used by the Dark Energy Survey analysis.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object.
        z (:obj:`float` or `array`): z value(s) where amplitude is evaluated.
        a1 (:obj:`float` or `array`): IA :math:`A_1` at input z values.
        a1delta (:obj:`float` or `array`): IA :math:`A_{1\\delta}` at input
            z values.
        a2 (:obj:`float` or `array`): IA :math:`A_2` at input z values.
        Om_m2_for_c2 (:obj:`bool`): True to use the Blazek et al. 2019
            convention of :math:`\\Omega_m^2` scaling.
        Om_m_fid (:obj:`float`): Value for Blazek et al. 2019 scaling.

    Returns:
        Tuple of IA bias parameters

        - c1 (:obj:`float` or `array`): IA :math:`C_1` at input z values.
        - c1delta (:obj:`float` or `array`): IA :math:`C_{1\\delta}` at
          input z values.
        - c2 (:obj:`float` or `array`): IA :math:`C_2` at input z values.
    """

    Om_m = cosmo['Omega_m']
    rho_crit = physical_constants.RHO_CRITICAL
    c1 = c1delta = c2 = None
    gz = cosmo.growth_factor(1./(1+z))

    if a1 is not None:
        c1 = -1*a1*5e-14*rho_crit*Om_m/gz

    if a1delta is not None:
        c1delta = -1*a1delta*5e-14*rho_crit*Om_m/gz

    if a2 is not None:
        if Om_m2_for_c2:  # Blazek2019 convention
            c2 = a2*5*5e-14*rho_crit*Om_m**2/(Om_m_fid*gz**2)
        else:  # DES convention
            c2 = a2*5*5e-14*rho_crit*Om_m/(gz**2)

    return c1, c1delta, c2


class PTTracer(CCLAutoRepr):
    """PTTracers contain the information necessary to describe the
    perturbative, non-linear inhomogeneities associated with
    different physical quantities.

    In essence their main function is to store a set of
    redshift-dependent functions (e.g. perturbation theory biases)
    needed in a perturbation theory framework to provide N-point
    correlations.
    """
    __repr_attrs__ = __eq_attrs__ = ('type', 'biases')

    def __init__(self):
        self.biases = {}
        pass

    @property
    @abstractmethod
    def type(self):
        """String defining tracer type (``'M'``, ``'NC'`` and
        ``'IA'`` supported).
        """

    def get_bias(self, bias_name, z):
        """Get the value of one of the bias functions at a given
        redshift.

        Args:
            bias_name (:obj:`str`): name of the bias function to return.
            z (:obj:`float` or `array`): redshift.

        Returns:
            (:obj:`float` or `array`): bias value at the input redshifts.
        """
        if bias_name not in self.biases:
            raise KeyError(f"Bias {bias_name} not included in this tracer")
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
    type = 'M'

    def __init__(self):
        self.biases = {}


class PTNumberCountsTracer(PTTracer):
    """:class:`PTTracer` representing number count fluctuations.
    This is described by 1st and 2nd-order biases and
    a tidal field bias. These are provided as floating
    point numbers or tuples of `(reshift,bias)` arrays.
    If a number is provided, a constant bias is assumed.
    If ``None``, a bias of zero is assumed.

    Args:
        b1 (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, b(z))`` giving the first-order
            bias.
        b2 (:obj:`float` or :obj:`tuple`): as above for the
            second-order bias.
        bs (:obj:`float` or :obj:`tuple`): as above for the
            tidal bias.
        b3nl (:obj:`float` or :obj:`tuple`): as above for the
            third-order bias.
        bk2 (:obj:`float` or :obj:`tuple`): as above for the
            non-local bias.
    """
    type = 'NC'

    def __init__(self, b1, b2=None, bs=None, b3nl=None, bk2=None):
        self.biases = {}

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2
        self.biases['b2'] = self._get_bias_function(b2)
        # Initialize bs
        self.biases['bs'] = self._get_bias_function(bs)
        # Initialize b3nl
        self.biases['b3nl'] = self._get_bias_function(b3nl)
        # Initialize bk2
        self.biases['bk2'] = self._get_bias_function(bk2)

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

    @property
    def b3nl(self):
        """Internal third-order bias function.
        """
        return self.biases['b3nl']

    @property
    def bk2(self):
        """Internal non-local bias function.
        """
        return self.biases['bk2']


class PTIntrinsicAlignmentTracer(PTTracer):
    """:class:`PTTracer` representing intrinsic alignments.
    This is described by 1st and 2nd-order alignment biases
    and an overdensity bias. These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If ``None``, a bias of zero is assumed.

    Args:
        c1 (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, c1(z))`` giving the first-order
            alignment bias :math:`C_1`.
        c2 (:obj:`float` or :obj:`tuple`): as above for the
            second-order alignment bias :math:`C_2`.
        cdelta (:obj:`float` or :obj:`tuple`): as above for the
            overdensity bias :math:`C_{1\\delta}`.
    """
    type = 'IA'

    def __init__(self, c1, c2=None, cdelta=None):

        self.biases = {}

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
