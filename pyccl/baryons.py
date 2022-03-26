from . import ccllib as lib
from .pyutils import check, deprecated
from .pk2d import Pk2D
from .base import unlock_instance
import numpy as np
import functools


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
    k_use = np.atleast_1d(k)
    status = 0
    fka, status = lib.bcm_model_fka_vec(cosmo.cosmo, a, k_use,
                                        len(k_use), status)
    check(status, cosmo)

    if np.ndim(k) == 0:
        fka = fka[0]
    return fka


def _bcm_correct_pk2d(cosmo, pk2d):
    """Apply the BCM model correction factor to a given power spectrum.
    This function operates directly onto the input Pk2D object.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        pk2d (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
    """
    if not isinstance(pk2d, Pk2D):
        raise TypeError("pk2d must be a Pk2D object")
    status = 0
    status = lib.bcm_correct(cosmo.cosmo, pk2d.psp, status)
    check(status, cosmo)


def baryon_correct(cosmo, model, pk2d):
    """Correct the power spectrum for baryons, given a model name string.
    This function operates on a copy of the input Pk2D object.

    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`):
            Cosmological parameters.
        model (str):
            Model to use.
        pk2d (:class:`~pyccl.pk2d.Pk2D`):
            Power spectrum.

    Returns:
        :class:`~pyccl.pk2d.Pk2D: a copy of the input `Pk2D` object with the
        baryon correction applied to it
    """
    if not isinstance(pk2d, Pk2D):
        raise TypeError("pk2d must be a Pk2D object")

    if model == "bcm":
        pk2d_new = pk2d.copy()
        _bcm_correct_pk2d(cosmo, pk2d_new)
    elif model in ["bacco", ]:  # other emulator names go in here
        from .emulator import PowerSpectrumEmulator
        emu = PowerSpectrumEmulator.from_name(model)()
        pk2d_new = emu.include_baryons(cosmo, pk2d)
    else:
        raise NotImplementedError(f"Baryon correction model {model} "
                                  "not recogized")
    return pk2d_new


@unlock_instance(mutate=True, argv=1)
@functools.wraps(_bcm_correct_pk2d)
@deprecated(new_function=baryon_correct)
def bcm_correct_pk2d(cosmo, pk2d):
    _bcm_correct_pk2d(cosmo, pk2d)
