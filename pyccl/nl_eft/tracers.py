__all__ = ("translate_IA_norm", "EFTTracer", "EFTMatterTracer",
           "EFTNumberCountsTracer", "EFTIntrinsicAlignmentTracer",)

from abc import abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from .. import CCLAutoRepr, physical_constants
from ..pyutils import _check_array_params


def translate_IA_norm(cosmo, *, z, a1=1.0):
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
    gz = cosmo.growth_factor(1./(1+z))

    if a1 is not None:
        b1g = -2*a1*5e-14*rho_crit*Om_m/gz

    return b1g


class EFTTracer(CCLAutoRepr):
    """EFTTracers contain the information necessary to describe the
    effective field theory-based non-linear inhomogeneities associated with
    different physical quantities.

    In essence their main function is to store a set of
    redshift-dependent functions (e.g. EFT biases)
    needed in the framework to provide N-point
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


class EFTMatterTracer(EFTTracer):
    """:class:`EFTTracer` representing matter fluctuations.
    """
    type = 'M'

    def __init__(self):
        self.biases = {}


class EFTNumberCountsTracer(EFTTracer):
    """:class:`EFTTracer` representing a scalar tracer (e.g.
    number counts). This is described by 1st, 2nd and 3rd-order 
    biases and a speed of sound bias. These are provided as floating
    point numbers or tuples of `(reshift,bias)` arrays.
    If a number is provided, a constant bias is assumed.
    If ``None``, a bias of zero is assumed. 

    Args:
        b1s (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of arrays ``(z, b(z))`` giving the first-order
            bias.
        b21s (:obj:`float` or :obj:`tuple`): as above for the
            second-order bias.
        b22s (:obj:`float` or :obj:`tuple`): as above for the
            second-order bias.
        b31s (:obj:`float` or :obj:`tuple`): as above for the
            third-order bias.
        bRs (:obj:`float` or :obj:`tuple`): as above for the
            speed of sound bias.
    """
    type = 'NC'

    def __init__(self, b1s, b21s=None, b22s=None, b31s=None, bRs=None):
        self.biases = {}

        # Initialize b1s
        self.biases['b1s'] = self._get_bias_function(b1s)
        # Initialize b21s
        self.biases['b21s'] = self._get_bias_function(b21s)
        # Initialize b22s
        self.biases['b22s'] = self._get_bias_function(b22s)
        # Initialize b31s
        self.biases['b31s'] = self._get_bias_function(b31s)
        # Initialize bRs
        self.biases['bRs'] = self._get_bias_function(bRs)

    @property
    def b1s(self):
        """Internal first-order bias function.
        """
        return self.biases['b1s']

    @property
    def b21s(self):
        """Internal second-order bias function.
        """
        return self.biases['b21s']

    @property
    def b22s(self):
        """Internal second-order bias function.
        """
        return self.biases['b22s']

    @property
    def b31s(self):
        """Internal third-order bias function.
        """
        return self.biases['b31s']

    @property
    def bRs(self):
        """Internal speed of sound bias function.
        """
        return self.biases['bRs']


class EFTIntrinsicAlignmentTracer(EFTTracer):
    """:class:`EFTTracer` representing a spin 2 tracer (e.g.,
    intrinsic alignments). This is described by 1st, 2nd and 
    3rd-order alignment biases and a speed of sound bias. 
    These bRge provided as floating point numbers or tuples of 
    (reshift,bias) bRgrays. If a number is provided, a constant 
    bias is assumed. If ``None``, a bias of zero is assumed.

    bRggs:
        b1g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the first-order
            alignment bias :math:`b_1^g`.
        b21g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the second-order
            alignment bias :math:`b_{2,1}^g`.
        b22g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the second-order
            alignment bias :math:`b_{2,2}^g`.
        b23g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the second-order
            alignment bias :math:`b_{2,3}^g`.
        b31g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the third-order
            alignment bias :math:`b_{3,1}^g`.
        b32g (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the third-order
            alignment bias :math:`b_{3,2}^g`.
        bRg (:obj:`float` or :obj:`tuple`): a single number or a
            tuple of bRgrays ``(z, c1(z))`` giving the speed of sound
            alignment bias :math:`b_R^g`.
    """
    type = 'IA'

    def __init__(self, b1g, b21g=None, b22g=None, b23g=None,
                 b31g=None, b32g=None, bRg=None):

        self.biases = {}

        # Initialize b1g
        self.biases['b1g'] = self._get_bias_function(b1g)
        # Initialize b21g
        self.biases['b21g'] = self._get_bias_function(b21g)
        # Initialize b22g
        self.biases['b22g'] = self._get_bias_function(b22g)
        # Initialize b23g
        self.biases['b23g'] = self._get_bias_function(b23g)
        # Initialize b31g
        self.biases['b31g'] = self._get_bias_function(b31g)
        # Initialize b32g
        self.biases['b32g'] = self._get_bias_function(b32g)
        # Initialize bRg
        self.biases['bRg'] = self._get_bias_function(bRg)

    @property
    def b1g(self):
        """Internal first-order bias function.
        """
        return self.biases['b1g']

    @property
    def b21g(self):
        """Internal second-order bias function.
        """
        return self.biases['b21g']

    @property
    def b22g(self):
        """Internal second-order bias function.
        """
        return self.biases['b22g']

    @property
    def b23g(self):
        """Internal second-order bias function.
        """
        return self.biases['b23g']

    @property
    def b31g(self):
        """Internal second-order bias function.
        """
        return self.biases['b31g']

    @property
    def b32g(self):
        """Internal second-order bias function.
        """
        return self.biases['b32g']

    @property
    def bRg(self):
        """Internal second-order bias function.
        """
        return self.biases['bRg']
