from .profiles import HaloProfile


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
