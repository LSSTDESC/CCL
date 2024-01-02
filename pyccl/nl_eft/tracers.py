__all__ = ("translate_IA_norm", "EFTTracer", "EFTMatterTracer",
           "EFTNumberCountsTracer", "EFTIntrinsicAlignmentTracer",)

# translating to EFT tracers
# utilizing BIAS, not bare bias coeffs of Vlah. et. al.18

from abc import abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from .. import CCLAutoRepr, physical_constants
from ..pyutils import _cheak_array_params


def translate_IA_EFT_norm(cosmo, *, z, a1=1.0, a21=None, a22=None,
                      a23=None, a31=None, a32=None, ak2=None,
                      Om_m2_for_a2=False, Om_m_fid=0.3, basis='NLA'):
    """
    Function to convert from :math:`A_{ia}` values to :math:`c_{ia}` values,
    for the intrinsic alignment bias parameters using the standard
    convention of `Blazek et al. 2019 <https://arxiv.org/abs/1708.09247>`_
    the variant used by the Dark Energy Survey analysis
    or the EFT parameterization 
    of 'Vlah et al. 2019 <https://arxiv.org/abs/1910.08085>'.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): cosmology object.
        z (:obj:`float` or `array`): z value(s) where amplitude is evaluated.
        a1 (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, a1(z))`` giving the first-order
            alignment bias A_1
        a21 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias.
            If specifying TATT, this is :math:`A_{1\\delta}`
        a22 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias.
            If specifying TATT, this is A_2
        a23 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias.
            If specifying TATT, this is b_t
        a31 (:obj:`float` or :obj:`tuple`): as above for
            third-order alignment bias.
        a32 (:obj:`float` or :obj:`tuple`): as above for
            third-order alignment bias.
        ak2 (:obj:`float` or :obj:`tuple`): as above for the
            k^2 bias
        Om_m2_for_a2 (:obj:`bool`): True to use the Blazek et al. 2019
            convention of :math:`\\Omega_m^2` scaling.
        Om_m_fid (:obj:`float`): Value for Blazek et al. 2019 scaling.
        basis (:obj:'string'): a flag denoting the internal interpretation
            of the parameters set by the user in the IA tracer. 
            Valid flags are 'EFT', 'TATT', and 'NLA'

    Returns:
        Tuple of IA bias parameters

        - a1 (:obj:`float` or `array`): IA :math:`C_1` at input z values.
        - a1delta (:obj:`float` or `array`): IA :math:`C_{1\\delta}` at
          input z values.
        - a2 (:obj:`float` or `array`): IA :math:`C_2` at input z values.
    """

    if basis is 'NLA' or basis is 'TATT':
        Om_m = cosmo['Omega_m']
        rho_crit = physical_constants.RHO_CRITICAL
        gz = cosmo.growth_factor(1./(1+z))
        zeros = np.zeros(len(z))
        c1 = c1delta = c2 = ct=zeros
        if a1 is not None:
            c1 = -1*a1*5e-14*rho_crit*Om_m/gz

        if a21 is not None:
            c1delta = -1*a21*5e-14*rho_crit*Om_m/gz

        if a22 is not None:
            if Om_m2_for_a2:  # Blazek2019 convention
                c2 = a22*5*5e-14*rho_crit*Om_m**2/(Om_m_fid*gz**2)
            else:  # DES convention
                c2 = a22*5*5e-14*rho_crit*Om_m/(gz**2)
        
        if a23 is not None:
            # check this normalizatoin
            ct = a23*5e-14*rho_crit*Om_m

        if c1!=zeros:
            a1 = c1
        #use transformation matrix here to convert to EFT parameters
        # as found in Bakx et al https://arxiv.org/pdf/2303.15565.pdf
        if basis is 'TATT':
            a21 = c2/3 + c1delta
            a22 = -8./7*ct - 2./3 * c2 + c1delta
            a23 = - 2./3 * c2 + c1delta
            if a21==zeros:
                a21 = None
            if a22==zeros:
                a22 = None
            if a23==zeros:
                a23 = None

    else:
        if a1 is not None:
            a1 *= 5e-14*rho_crit*Om_m
        if a21 is not None:
            a21 *= 5e-14*rho_crit*Om_m
        if a22 is not None:
            a22 *= 5e-14*rho_crit*Om_m
        if a23 is not None:
            a23 *= 5e-14*rho_crit*Om_m
        if a31 is not None:
            a31 *= 5e-14*rho_crit*Om_m
        if a32 is not None:
            a32 *= 5e-14*rho_crit*Om_m
        if ak2 is not None:
            ak2 *= 5e-14*rho_crit*Om_m

    return a1, a21, a22, a23, a31, a32, ak2


class EFTTracer(CCLAutoRepr):
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
            z, b = _cheak_array_params(b)
            return interp1d(z, b, bounds_error=False,
                            fill_value=b[-1])


class EFTMatterTracer(PTTracer):
    """:class:`EFTTracer` representing matter fluctuations.
    """
    type = 'M'

    def __init__(self):
        self.biases = {}


class EFTNumberCountsTracer(PTTracer):
    """:class:`EFTTracer` representing number count fluctuations.
    This is described by 1st and 2nd-order biases and
    a tidal field bias. These are provided as floating
    point numbers or tuples of `(reshift,bias)` arrays.
    If a number is provided, a constant bias is assumed.
    If ``None``, a bias of zero is assumed.

    Args:
        b1 (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, b(z))`` giving the first-order
            bias.
        b21 (:obj:`float` or :obj:`tuple`): as above for
            second-order bias.
        b22 (:obj:`float` or :obj:`tuple`): as above for
            second-order bias.
        b31 (:obj:`float` or :obj:`tuple`): as above for the
            third-order bias.
        bk2 (:obj:`float` or :obj:`tuple`): as above for the
            non-local bias.
    """
    type = 'NC'

    def __init__(self, b1, b2=None, bs=None, b31=None, bk2=None):
        self.biases = {}

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2,1
        self.biases['b2,1'] = self._get_bias_function(b21)
        # Initialize b2,2
        self.biases['b2,2'] = self._get_bias_function(b22)
        # Initialize b3,1
        self.biases['b3,1'] = self._get_bias_function(b31)
        # Initialize bk2
        self.biases['bk2'] = self._get_bias_function(bk2)

    @property
    def b1(self):
        """Internal first-order bias function.
        """
        return self.biases['b1']

    @property
    def b21(self):
        """Internal second-order bias function.
        """
        return self.biases['b2,1']

    @property
    def b22(self):
        """Internal tidal bias function.
        """
        return self.biases['b2,2']

    @property
    def b31(self):
        """Internal third-order bias function.
        """
        return self.biases['b3,1']

    @property
    def bk2(self):
        """Internal non-local bias function.
        """
        return self.biases['bk2']


class EFTIntrinsicAlignmentTracer(PTTracer):
    """:class:`EFTTracer` representing intrinsic alignments.
    This is described by 1st and 2nd-order alignment biases
    and an overdensity bias. These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If ``None``, a bias of zero is assumed.

    Args:
        a1 (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, a1(z))`` giving the first-order
            alignment bias A_1
        a21 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias. In TATT, this is b_TA
        a22 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias. In TATT, this is A_2
        a23 (:obj:`float` or :obj:`tuple`): as above for
            second-order alignment bias. In TATT, this is b_t
        a31 (:obj:`float` or :obj:`tuple`): as above for
            third-order alignment bias.
        a32 (:obj:`float` or :obj:`tuple`): as above for
            third-order alignment bias.
        ak2 (:obj:`float` or :obj:`tuple`): as above for the
            k^2 bias
        basis (:obj:'string'): a flag denoting the internal interpretation
            of the parameters set by the user. 
            Valid flags are 'EFT', 'TATT', and 'NLA'
    """
    type = 'IA'
    bases = ['EFT', 'TATT', 'NLA']

    def __init__(self, a1, a21=None, a22=None, a23=None, a31=None, a32=None, ak2=None, basis = 'NLA'):

        if basis not in bases:
            raise ValueError(f"Unknown IA prescription {basis}")

        self.biases = {}
        # Initialize a1
        self.biases['a1'] = self._get_bias_function(a1)
        if(basis is not "NLA"):
            # Initialize a2,1
            self.biases['a2,1'] = self._get_bias_function(a21)
            # Initialize a2,2
            self.biases['a2,2'] = self._get_bias_function(a22)
            if basis is not 'TATT':
                # Initialize a2,3
                self.biases['a2,3'] = self._get_bias_function(a23)
                # Initialize a3,1
                self.biases['a3,1'] = self._get_bias_function(a31)
                # Initialize a3,2
                self.biases['a3,2'] = self._get_bias_function(a32)
                # Initialize ak2
                self.biases['ak2'] = self._get_bias_function(ak2)

    @property
    def a1(self):
        """Internal first-order bias function.
        """
        return self.biases['a1']

    @property
    def a21(self):
        """Internal second-order bias function.
        """
        return self.biases['a21']

    @property
    def a22(self):
        """Internal second-order bias function.
        """
        return self.biases['a22']

    @property
    def a23(self):
        """Internal second-order bias function.
        """
        return self.biases['a23']

    @property
    def a31(self):
        """Internal third-order bias function.
        """
        return self.biases['a31']

    @property
    def a32(self):
        """Internal third-order bias function.
        """
        return self.biases['a32']

    @property
    def ak2(self):
        """Internal k^2 bias function.
        """
        return self.biases['ak2']

