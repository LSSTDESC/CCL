from .profiles import HaloProfile, HaloProfileHOD


class Profile2pt(object):
    """ This class implements the 1-halo 2-point correlator between two
    halo profiles. In the simplest case, this is just
    the product of both profiles in Fourier space.
    More complicated cases should be implemented by subclassing
    this class and overloading the :meth:`~Profile2pt.fourier_2pt`
    method.
    """
    def __init__(self):
        pass

    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment between
        two profiles:

        .. math::
           \\langle\\rho_1(k)\\rho_2(k)\\rangle.

        Args:
            prof (:class:`~pyccl.halos.profiles.HaloProfile`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation, and `prof` will be used as `prof2`.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof2, HaloProfile):
                raise TypeError("prof2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof2.fourier(cosmo, k, M, a, mass_def=mass_def)

        return uk1 * uk2


class Profile2ptHOD(Profile2pt):
    """ This class implements the Fourier-space 1-halo 2-point
    correlator for the HOD profile.

    .. math::
       \\langle n_g^2(k)|M,a\\rangle = \\bar{N}_c(M,a)
       \\left[2f_c(a)\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)+
       (\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a))^2\\right],

    where all quantities are described in the documentation of
    :class:`~pyccl.halos.profiles.HaloProfileHOD`.
    """
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment for the HOD
        profile.

        Args:
            prof (:class:`~pyccl.halos.profiles.HaloProfileHOD`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfileHOD):
            raise TypeError("prof must be of type `HaloProfileHOD`")

        if prof2 is not None:
            if prof2 is not prof:
                raise ValueError("prof2 must be the same as prof")

        return prof._fourier_variance(cosmo, k, M, a, mass_def)


class Profile2ptR(Profile2pt):
    """ This class inherits from the 1-halo 2-point correlator between two
    halo profiles. Sometimes we need not assume that the 1-halo terms are
    fully correlated. We introduce a term `r_corr` which scales the product
    of the fourier halo profiles by `(1+r_corr)`. In the trivial case where
    `r_corr=0` this returns the 2-point correlator of the parent class.
    Example usecases can be found in arXiv:1909.09102 and arXiv:2102.07701.
    """
    def __init__(self, r_corr=0.):
        self.r_corr = r_corr

    def update_parameters(self, r_corr=None):
        """ Update any of the parameters associated with this 1-halo
        2-point correlator. Any parameter set to `None` won't be updated.
        """
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(self, prof, cosmo, k, M, a, r_corr=None,
                    prof2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment between
        two profiles:

        Args:
            prof (:class:`~pyccl.halos.profiles.HaloProfileHOD`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            r_corr (float): scale the correlation through `1-r_corr`.
                If `None` it will default to the value passed during
                instantiation of this class. Note, that if you require
                to update `r_corr` for the instance created, you need to
                do it through `update_parameters`.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation. Note that only auto-correlations
                are allowed in this case.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof2, HaloProfile):
                raise TypeError("prof2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof2.fourier(cosmo, k, M, a, mass_def=mass_def)

        if r_corr is None:
            r_corr = self.r_corr
        return uk1 * uk2 * (1 + r_corr)
