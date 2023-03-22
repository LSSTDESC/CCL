from .base import unlock_instance
from .baryons import BaryonsSchneider15
from .pyutils import deprecated


@deprecated(BaryonsSchneider15)
def bcm_model_fka(cosmo, k, a):
    """The BCM model correction factor for baryons.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: The correction factor is applied multiplicatively so that
              :math:`P_{\\rm corrected}(k, a) = P(k, a)\\, f_{\\rm bcm}(k, a)`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Correction factor to apply to the power spectrum.
    """
    bcm = BaryonsSchneider15(log10Mc=cosmo['bcm_log10Mc'],
                             eta_b=cosmo['bcm_etab'],
                             k_s=cosmo['bcm_ks'])
    return bcm.boost_factor(cosmo, k, a)


@deprecated(BaryonsSchneider15)
@unlock_instance(mutate=True, argv=1)
def bcm_correct_pk2d(cosmo, pk2d):
    """Apply the BCM model correction factor to a given power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        pk2d (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
    """

    bcm = BaryonsSchneider15(log10Mc=cosmo['bcm_log10Mc'],
                             eta_b=cosmo['bcm_etab'],
                             k_s=cosmo['bcm_ks'])
    bcm.include_baryonic_effects(cosmo, pk2d, in_place=True)
