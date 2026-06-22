__all__ = ("Profile2pt", "Profile2ptHOD", "Profile2ptCIB",)

from .. import CCLAutoRepr
from . import HaloProfile, HaloProfileHOD, HaloProfileCIBShang12


class Profile2pt(CCLAutoRepr):
    """ This class implements the 1-halo 2-point correlator between
    two halo profiles.

    .. math::
        \\langle u_1(k) u_2(k) \\rangle.

    In the simplest case the second-order cumulant is just the product
    of the individual Fourier-space profiles. More complicated cases
    are implemented via the parameters of this class.

    Args:
        r_corr (:obj:`float`):
            Tuning knob for the 1-halo 2-point correlation.
            Scale the correlation by :math:`(1+\\rho_{u_1, u_2})`.
            This is useful when the individual 1-halo terms
            are not fully correlated. Example usecases can be found
            in ``arXiv:1909.09102`` and ``arXiv:2102.07701``.
            Defaults to ``r_corr=0``, returning simply the product
            of the fourier profiles.
    """
    __repr_attrs__ = __eq_attrs__ = ("r_corr",)

    def __init__(self, *, r_corr=0.):
        self.r_corr = r_corr

    def update_parameters(self, *, r_corr=None):
        """ Update any of the parameters associated with this 1-halo
        2-point correlator. Any parameter set to ``None`` won't be updated.
        """
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True):
        """ Return the Fourier-space two-point moment between
        two profiles.

        .. math::
           (1+\\rho_{u_1,u_2})\\langle u_1(k)\\rangle\\langle u_2(k) \\rangle

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                a Cosmology object.
            k (:obj:`float` or `array`):
                comoving wavenumber in Mpc^-1.
            M (:obj:`float` or `array`):
                halo mass in units of M_sun.
            a (:obj:`float`):
                scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`):
                halo profile for which the second-order moment
                is desired.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If ``None``, the assumption is that you want
                an auto-correlation, and `prof` will be used as `prof2`.
            diag (bool): If True, both halo profiles depend on the same k. If
            False, they will depend on k and k', respectively. The output
            dimension will change to `(N_M, N_k, N_k)`.Default True.

        Returns:
            (:obj:`float` or `array`): second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively, if diag is True. If False, the array will have
            dimension `(N_M, N_k, N_k)`, with k' corresponding to the second
            axis of the array.. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if prof2 is None:
            prof2 = prof

        HP = HaloProfile
        if not (isinstance(prof, HP) and isinstance(prof2, HP)):
            raise TypeError("prof and prof2 must be HaloProfile")

        uk1 = prof.fourier(cosmo, k, M, a)

        if prof == prof2:
            uk2 = uk1
        else:
            uk2 = prof2.fourier(cosmo, k, M, a)

        # TODO: This should be implemented in _fourier_variance
        if (diag is True) or (isinstance(k, float)):
            output = uk1 * uk2 * (1 + self.r_corr)
        elif isinstance(M, float):
            output = uk1[None, :] * uk2[:, None] * (1 + self.r_corr)
        else:
            output = uk1[:, None, :] * uk2[:, :, None] * (1 + self.r_corr)

        return output


class Profile2ptHOD(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the HOD profile.

    .. math::
       \\langle n_g^2(k)|M,a\\rangle = \\bar{N}_c(M,a)
       \\left[2f_c(a)\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)+
       (\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a))^2\\right],

    where all quantities are described in the documentation of
    :class:`~pyccl.halos.profiles.hod.HaloProfileHOD`.
    """

    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True):
        """ Returns the Fourier-space two-point moment for the HOD
        profile.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.hod.HaloProfileHOD`):
                halo profile for which the second-order moment
                is desired.
            prof2 (:class:`~pyccl.halos.profiles.hod.HaloProfileHOD` or :obj:`None`):
                second halo profile for which the second-order moment
                is desired. If ``None``, the assumption is that you want
                an auto-correlation. If the profiles are different we assume disjoint 
                tracers and revert back to the usual Fourier-space two-point moment 
                (see Profile2pt).
            diag (bool): If True, both halo profiles depend on the same k. If
                False, they will depend on k and k', respectively and we will
                approximate <uk uk'> to <uk><uk'>. The output dimension will
                change to `(N_M, N_k, N_k)`. Default True.

        Returns:
            (:obj:`float` or `array`): second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively, if diag is True. If False, the array will have
            dimension `(N_M, N_k, N_k)`, with k' corresponding to the second
            axis of the array.. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """ # noqa
        if prof2 is None:
            prof2 = prof

        # If the profiles are different assume disjoint tracers
        if prof != prof2:
            uk1 = prof.fourier(cosmo, k, M, a)
            uk2 = prof2.fourier(cosmo, k, M, a)

        HOD = HaloProfileHOD
        if not (isinstance(prof, HOD) and isinstance(prof2, HOD)):
            raise TypeError("prof and prof2 must be HaloProfileHOD")

        # TODO: This should be implemented in _fourier_variance
        if (diag is True) or (isinstance(k, float)):
            if prof == prof2:
                output = prof._fourier_variance(cosmo, k, M, a)
            else:
                output = uk1 * uk2 * (1 + self.r_corr)
        elif isinstance(M, float):
            if prof == prof2:
                uk1 = prof.fourier(cosmo, k, M, a)
                output = uk1[None, :] * uk1[:, None] * (1 + self.r_corr)
            else:
                output = uk1[None, :] * uk2[:, None] * (1 + self.r_corr)
        else:
            if prof == prof2:
                uk1 = prof.fourier(cosmo, k, M, a)
                output = uk1[:, None, :] * uk1[:, :, None] * (1 + self.r_corr)
            else:
                output = uk1[:, None, :] * uk2[:, :, None] * (1 + self.r_corr)

        return output


class Profile2ptCIB(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the CIB profile. It follows closely the
    implementation of the equivalent HOD quantity
    (see :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`
    and Eq. 15 of `McCarthy & Madhavacheril
    <https://arxiv.org/abs/2010.16405>`_).
    """

    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True):
        """ Returns the Fourier-space two-point moment for the CIB
        profile.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
            M (:obj:`float` or `array`): halo mass in units of M_sun.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.cib_shang12.HaloProfileCIBShang12`):
                halo profile for which the second-order moment
                is desired.
            prof2 (:class:`~pyccl.halos.profiles.cib_shang12.HaloProfileCIBShang12`):
                second halo profile for which the second-order moment
                is desired. If ``None``, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            diag (bool): If True, both halo profiles depend on the same k. If
                False, they will depend on k and k', respectively and we will
                approximate <uk uk'> to <uk><uk'>. The output dimension will
                change to `(N_M, N_k, N_k)`. Default True.

        Returns:
            (:obj:`float` or `array`): second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively, if diag is True. If False, the array will have
            dimension `(N_M, N_k, N_k)`, with k' corresponding to the second
            axis of the array. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """ # noqa
        if prof2 is None:
            prof2 = prof

        Shang12 = HaloProfileCIBShang12
        if not (isinstance(prof, Shang12) and isinstance(prof2, Shang12)):
            raise TypeError("prof and prof2 must be HaloProfileCIB")

        # TODO: This should be implemented in _fourier_variance
        if (diag is True) or (isinstance(k, float)):
            output = prof._fourier_variance(cosmo, k, M, a, nu_other=prof2.nu)
        elif isinstance(M, float):
            uk1 = prof.fourier(cosmo, k, M, a)
            uk2 = prof2.fourier(cosmo, k, M, a)
            output = uk1[None, :] * uk2[:, None] * (1 + self.r_corr)
        else:
            uk1 = prof.fourier(cosmo, k, M, a)
            uk2 = prof2.fourier(cosmo, k, M, a)
            output = uk1[:, None, :] * uk2[:, :, None] * (1 + self.r_corr)

        return output
