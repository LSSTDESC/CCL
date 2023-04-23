__all__ = ("FFTLogParams",)


class FFTLogParams:
    """Objects of this class store the FFTLog accuracy parameters.

    Stored in instances of :obj:`~pyccl.halos.HaloProfile`.
    """
    padding_lo_fftlog = 0.1   # | Anti-aliasing: multiply the lower boundary.
    padding_hi_fftlog = 10.   # |                multiply the upper boundary.

    n_per_decade = 100        # Samples per decade for the Hankel transforms.
    extrapol = "linx_liny"     # Extrapolation type.

    padding_lo_extra = 0.1    # Padding for the intermediate step of a double
    padding_hi_extra = 10.    # transform. Doesn't have to be as precise.
    large_padding_2D = False  # If True, high precision intermediate transform.

    plaw_fourier = -1.5       # Real <--> Fourier transforms.
    plaw_projected = -1.0     # 2D projected & cumulative density profiles.

    @property
    def params(self):
        return ["padding_lo_fftlog", "padding_hi_fftlog", "n_per_decade",
                "extrapol", "padding_lo_extra", "padding_hi_extra",
                "large_padding_2D", "plaw_fourier", "plaw_projected"]

    def to_dict(self):
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
            return True
        if type(self) != type(other):
            return False
        return self.to_dict() == other.to_dict()

    def update_parameters(self, **kwargs):
        r"""Update the precision of FFTLog for the Hankel transforms.

        Arguments
        ---------
        padding_lo_fftlog, padding_hi_fftlog : float
            Multiply the lower and upper boundary of the input range
            to avoid aliasing. The defaults are 0.1 and 10.0, respectively.
        n_per_decade : float
            Samples per decade for the Hankel transforms.
            The default is 100.
        extrapol : {'linx_liny', 'linx_logy'}
            Extrapolation type when FFTLog has narrower output support.
            The default is 'linx_liny'.
        padding_lo_extra, padding_hi_extra : float
            Padding for the intermediate step of a double Hankel transform.
            Used to compute the 2D projected profile and the 2D cumulative
            density, where the first transform goes from 3D real space to
            Fourier, then from Fourier to 2D real space. Usually, it doesn't
            have to be as precise as ``padding_xx_fftlog``.
            The defaults are 0.1 and 10.0, respectively.
        large_padding_2D : bool
            Override ``padding_xx_extra`` in the intermediate transform,
            and use ``padding_xx_fftlog``. The default is False.
        plaw_fourier, plaw_projected : float
            FFTLog pre-whitens its arguments (makes them flatter) to avoid
            aliasing. The ``plaw`` parameters describe the tilt of the profile,
            :math:`P(r) \sim r^{\rm tilt}`, between real and Fourier
            transforms, and between 2D projected and cumulative density,
            respectively. Subclasses of ``HaloProfile`` may obtain finer
            control via ``_get_plaw_[fourier | projected]``, and some level of
            experimentation with these parameters is recommended.
            The defaults are -1.5 and -1.0, respectively.
        """
        for name, value in kwargs.items():
            if name not in self.params:
                raise AttributeError(f"Parameter {name} does not exist.")
            object.__setattr__(self, name, value)
