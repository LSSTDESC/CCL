__all__ = ("FFTLogParams",)


class FFTLogParams:
    """Objects of this class store the FFTLog accuracy parameters.
    See documentation in :meth:`update_parameters` for a full
    description of all allowed parameters.
    """
    #: Anti-aliasing. Factor mulitplying the lower boundary.
    padding_lo_fftlog = 0.1
    #: Anti-aliasing. Factor mulitplying the upper boundary.
    padding_hi_fftlog = 10.

    #: Samples per decade for the Hankel transforms.
    n_per_decade = 100
    #: Extrapolation type (`linx_liny`, `linx_logy` etc.).
    extrapol = "linx_liny"

    #: Padding for intermediate transforms (lower bound).
    padding_lo_extra = 0.1
    #: Padding for intermediate transforms (upper bound).
    padding_hi_extra = 10.
    #: If True, high precision intermediate transforms.
    large_padding_2D = False

    #: Power law index used to prewhiten data before transform.
    plaw_fourier = -1.5
    #: Pre-whitening power law index for 2D and cumulative profiles.
    plaw_projected = -1.0

    @property
    def params(self):
        return ["padding_lo_fftlog", "padding_hi_fftlog", "n_per_decade",
                "extrapol", "padding_lo_extra", "padding_hi_extra",
                "large_padding_2D", "plaw_fourier", "plaw_projected"]

    def to_dict(self):
        """ Returns a dictionary containing this object's parameters.
        """
        return {param: getattr(self, param) for param in self.params}

    def __getitem__(self, name):
        return getattr(self, name)

    def __setattr__(self, name, value):
        raise AttributeError("FFTLogParams can only be updated via "
                             "`updated_parameters`.")

    def __repr__(self):
        return repr(self.to_dict())

    def __eq__(self, other):
        if self is other:
            True
        if type(self) is not type(other):
            return False
        return self.to_dict() == other.to_dict()

    def update_parameters(self, **kwargs):
        """Update the precision of FFTLog for the Hankel transforms.

        Arguments
        ---------
        padding_lo_fftlog: :obj:`float`
            Factor by which the minimum scale is multiplied to avoid
            aliasing. Default: 0.1.
        padding_hi_fftlog: :obj:`float`
            Factor by which the maximum scale is multiplied to avoid
            aliasing. Default: 10.
        n_per_decade : :obj:`float`
            Samples per decade for the Hankel transforms. Default: 100.
        extrapol : {'linx_liny', 'linx_logy'}
            Extrapolation type when FFTLog has narrower output support.
            Default ``'linx_liny'``.
        padding_lo_extra: :obj:`float`
            Additional minimum scale padding for double Hankel transforms,
            used when computing 2D projected and cumulative profiles. In
            these, the first transform goes from 3D real space to
            Fourier, and the second transform goes from Fourier to 2D
            real space.
            Default: 0.1.
        padding_hi_extra: :obj:`float`
            As ``padding_lo_extra`` for the maximum scale.
            Default: 10.
        large_padding_2D : :obj:`bool`
            Override ``padding_xx_extra`` in the intermediate transform,
            and use ``padding_xx_fftlog``. The default is False.
        plaw_fourier: :obj:`float`
            FFTLog pre-whitens its arguments (makes them flatter) to avoid
            aliasing. The ``plaw_fourier`` parameter describes the tilt of
            the profile, :math:`P(r) \\propto r^{\\mathrm{tilt}}`, for
            standard 3D transforms. Default: -1.5
        plaw_fourier_projected: :obj:`float`
            As ``plaw_fourier`` for 2D transforms (when computing 2D
            projected or cumulative profiles. Default: -1.0.
        """
        for name, value in kwargs.items():
            if name not in self.params:
                raise AttributeError(f"Parameter {name} does not exist.")
            object.__setattr__(self, name, value)
