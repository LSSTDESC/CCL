from .profiles import HaloProfile, HaloProfileHOD


class Profile2pt(object):
    """ This class implements the 1-halo 2-point correlator between
    two halo profiles.

    .. math::
        \\langle u_1(k) u_2(k) \\rangle.

    In the simplest case the second-order cumulant is just the product
    of the individual Fourier-space profiles. More complicated cases
    are implemented via the parameters of this class.

    Args:
        r_corr (float):
            Tuning knob for the 1-halo 2-point correlation.
            Scale the correlation by :math:`(1+\\rho_{u_1, u_2})`.
            This is useful when the individual 1-halo terms
            are not fully correlated. Example usecases can be found
            in ``arXiv:1909.09102`` and ``arXiv:2102.07701``.
            Defaults to ``r_corr=0``, returning simply the product
            of the fourier profiles.

    """
    def __init__(self, r_corr=0.):
        self.r_corr = r_corr

    def update_parameters(self, r_corr=None):
        """ Update any of the parameters associated with this 1-halo
        2-point correlator. Any parameter set to `None` won't be updated.
        """
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        """ Return the Fourier-space two-point moment between
        two profiles.

        .. math::
           (1+\\rho_{u_1,u_2})\\langle u_1(k)\\rangle\\langle u_2(k) \\rangle

        Args:
            prof (:class:`~pyccl.halos.profiles.HaloProfile`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`):
                a Cosmology object.
            k (float or array_like):
                comoving wavenumber in Mpc^-1.
            M (float or array_like):
                halo mass in units of M_sun.
            a (float):
                scale factor.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation, and `prof` will be used as `prof2`.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

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

        return uk1 * uk2 * (1 + self.r_corr)


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
