from . import ccllib as lib
from .pyutils import _vectorize_fn2


def bcm_model_fka(cosmo, k, a):
    """The BCM model correction factor for baryons.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: The correction factor is applied multiplicatively so that
              `P_corrected(k, a) = P(k, a) * factor(k, a)`.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Correction factor to apply to the power spectrum.
    """
    return _vectorize_fn2(lib.bcm_model_fka,
                          lib.bcm_model_fka_vec, cosmo, k, a)
