"""
This script contains methods for ~pyccl.pk2d.Pk2D.
`cls` is the `Pk2D` class, while `self` is an instance of the `Pk2D` class.
"""
import warnings
import numpy as np
from . import ccllib as lib
from .emulator import PowerSpectrumEmulator
from .pyutils import check, CCLWarning, deprecated


class _Pk2D_descriptor(object):
    """
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, base):
        if instance is None:
            warnings.warn("Use of the power spectrum as an argument "
                          f"is deprecated in {self.func.__name__}. "
                          "Use the instance method instead.", CCLWarning)
            this = base
        else:
            this = instance

        def new_func(*args, **kwargs):
            return self.func(this, *args, **kwargs)
        new_func.__name__ = self.func.__name__
        new_func.__doc__ = self.func.__doc__

        return new_func


def from_model(cls, cosmo, model):
    """`Pk2D` constructor returning the power spectrum associated with
    a given numerical model.

    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`)
            A Cosmology object.
        model (:obj:`str`)
            model to use. These models allowed:
              - `'bbks'` (Bardeen et al. ApJ 304 (1986) 15)
              - `'eisenstein_hu'` (Eisenstein & Hu astro-ph/9709112)
              - `'eisenstein_hu_nowiggles'` (Eisenstein & Hu astro-ph/9709112)
              - `'emu'` (arXiv:1508.02654).

    Returns:
        :class:`~pyccl.pk2d.Pk2D`
            The power spectrum of the input model.
    """
    if model in ['bacco', ]:  # other emulators go in here
        return PowerSpectrumEmulator.get_pk_linear(cosmo, model)

    pk2d = cls(empty=True)
    status = 0
    if model == 'bbks':
        cosmo.compute_growth()
        ret = lib.compute_linpower_bbks(cosmo.cosmo, status)
    elif model == 'eisenstein_hu':
        cosmo.compute_growth()
        ret = lib.compute_linpower_eh(cosmo.cosmo, 1, status)
    elif model == 'eisenstein_hu_nowiggles':
        cosmo.compute_growth()
        ret = lib.compute_linpower_eh(cosmo.cosmo, 0, status)
    elif model == 'emu':
        ret = lib.compute_power_emu(cosmo.cosmo, status)
    else:
        raise ValueError("Unknown model %s " % model)

    if np.ndim(ret) == 0:
        status = ret
    else:
        pk2d.psp, status = ret

    check(status, cosmo)
    pk2d.has_psp = True
    return pk2d


@deprecated(new_function=from_model)
def pk_from_model(cls, cosmo, model):
    """`Pk2D` constructor returning the power spectrum associated with
    a given numerical model.

    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`)
            A Cosmology object.
        model (:obj:`str`)
            model to use. These models allowed:
              - `'bbks'` (Bardeen et al. ApJ 304 (1986) 15)
              - `'eisenstein_hu'` (Eisenstein & Hu astro-ph/9709112)
              - `'eisenstein_hu_nowiggles'` (Eisenstein & Hu astro-ph/9709112)
              - `'emu'` (arXiv:1508.02654).

    Returns:
        :class:`~pyccl.pk2d.Pk2D`
            The power spectrum of the input model.
    """
    return from_model(cls, cosmo, model)


def apply_halofit(self, cosmo, pk_linear=None):
    """Pk2D constructor that applies the "HALOFIT" transformation of
    Takahashi et al. 2012 (arXiv:1208.2701) on an input linear
    power spectrum in `pk_linear`.

    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`)
            A Cosmology object.
        pk_linear (:class:`Pk2D`)
            A :class:`Pk2D` object containing the linear power spectrum
            to transform. This argument is deprecated and will be removed
            in a future release. Use the instance method instead.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`
            A copy of the input power spectrum where the HALOFIT
            transformation has been applied.
    """
    if pk_linear is not None:
        self = pk_linear

    from .pk2d import Pk2D
    pk2d = Pk2D(empty=True)
    status = 0
    ret = lib.apply_halofit(cosmo.cosmo, self.psp, status)
    if np.ndim(ret) == 0:
        status = ret
    else:
        pk2d.psp, status = ret
    check(status, cosmo)
    pk2d.has_psp = True
    return pk2d


def apply_nonlin_model(self, cosmo, model, pk_linear=None):
    """Pk2D constructor that applies a non-linear model
    to a linear power spectrum.

    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`)
            A Cosmology object.
        model (str)
            Model to use.
        pk_linear (:class:`Pk2D`)
            A :class:`Pk2D` object containing the linear power spectrum
            to transform. This argument is deprecated and will be removed
            in a future release. Use the instance method instead.

    Returns:
        :class:`Pk2D`
            A copy of the input power spectrum where the nonlinear model
            has been applied.
    """
    if pk_linear is not None:
        self = pk_linear

    if model == "halofit":
        pk2d_new = self.apply_halofit(cosmo)
    # elif model in ["bacco", ]:  # other emulator names go in here
    #     from .boltzmann import PowerSpectrumEmulator as PSE
    #     pk2d_new = PSE.apply_nonlin_model(cosmo, model, self)
    return pk2d_new


def include_baryons(self, cosmo, model, pk_nonlin=None):
    """Pk2D constructor that applies a correction for baryons to
    a non-linear power spectrum.
    Arguments:
        cosmo (:class:`~pyccl.core.Cosmology`)
            A Cosmology object.
        model (str)
            Model to use.
        pk_nonlin (:class:`Pk2D`)
            A :class:`Pk2D` object containing the non-linear power spectrum
            to transform. This argument is deprecated and will be removed
            in a future release. Use the instance method instead.

    Returns:
        :class:`Pk2D`
            A copy of the input power spectrum where the baryon correction
            has been applied.
    """
    # As in `apply_nonlin_model`, use of `pk_nonlin` should be deprecated.
    if pk_nonlin is not None:
        self = pk_nonlin

    return cosmo.baryon_correct(model, self)
