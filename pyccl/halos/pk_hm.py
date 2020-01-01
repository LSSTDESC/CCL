from .profiles import HaloProfile


class ProfileCovar(object):
    def __init__(self):
        pass

    def fourier_covar(self, prof, cosmo, k, M, a, prof_2=None, mass_def=None):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof_2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof_2, HaloProfile):
                raise TypeError("prof_2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof_2.fourier(cosmo, k, M, a, mass_def=mass_def)

        return uk1 * uk2
