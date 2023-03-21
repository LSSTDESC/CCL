from ..concentration import Concentration
from .profile_base import HaloProfile
import numpy as np
from scipy.special import sici, erf


__all__ = ("HaloProfileHOD",)


class HaloProfileHOD(HaloProfile):
    """ A generic halo occupation distribution (HOD)
    profile describing the number density of galaxies
    as a function of halo mass.

    The parametrization for the mean profile is:

    .. math::
       \\langle n_g(r)|M,a\\rangle = \\bar{N}_c(M,a)
       \\left[f_c(a)+\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)\\right]

    where :math:`\\bar{N}_c` and :math:`\\bar{N}_s` are the
    mean number of central and satellite galaxies respectively,
    :math:`f_c` is the observed fraction of central galaxies, and
    :math:`u_{\\rm sat}(r|M,a)` is the distribution of satellites
    as a function of distance to the halo centre.

    These quantities are parametrized as follows:

    .. math::
       \\bar{N}_c(M,a)=\\frac{1}{2}\\left[1+{\\rm erf}
       \\left(\\frac{\\log(M/M_{\\rm min})}{\\sigma_{{\\rm ln}M}}
       \\right)\\right]

    .. math::
       \\bar{N}_s(M,a)=\\Theta(M-M_0)\\left(\\frac{M-M_0}{M_1}
       \\right)^\\alpha

    .. math::
       u_s(r|M,a)\\propto\\frac{\\Theta(r_{\\rm max}-r)}
       {(r/r_g)(1+r/r_g)^2}

    Where :math:`\\Theta(x)` is the Heaviside step function,
    and the proportionality constant in the last equation is
    such that the volume integral of :math:`u_s` is 1. The
    radius :math:`r_g` is related to the NFW scale radius :math:`r_s`
    through :math:`r_g=\\beta_g\\,r_s`, and the radius
    :math:`r_{\\rm max}` is related to the overdensity radius
    :math:`r_\\Delta` as :math:`r_{\\rm max}=\\beta_{\\rm max}r_\\Delta`.
    The scale radius is related to the comoving overdensity halo
    radius via :math:`R_\\Delta(M) = c(M)\\,r_s`.

    All the quantities :math:`\\log_{10}M_{\\rm min}`,
    :math:`\\log_{10}M_0`, :math:`\\log_{10}M_1`,
    :math:`\\sigma_{{\\rm ln}M}`, :math:`f_c`, :math:`\\alpha`,
    :math:`\\beta_g` and :math:`\\beta_{\\rm max}` are
    time-dependent via a linear expansion around a pivot scale
    factor :math:`a_*` with an offset (:math:`X_0`) and a tilt
    parameter (:math:`X_p`):

    .. math::
       X(a) = X_0 + X_p\\,(a-a_*).

    This definition of the HOD profile draws from several papers
    in the literature, including: astro-ph/0408564, arXiv:1706.05422
    and arXiv:1912.08209. The default values used here are roughly
    compatible with those found in the latter paper.

    See :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD`) for a
    description of the Fourier-space two-point correlator of the
    HOD profile.

    Args:
        c_M_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        lMmin_0 (float): offset parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        lMmin_p (float): tilt parameter for
            :math:`\\log_{10}M_{\\rm min}`.
        siglM_0 (float): offset parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        siglM_p (float): tilt parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        lM0_0 (float): offset parameter for
            :math:`\\log_{10}M_0`.
        lM0_p (float): tilt parameter for
            :math:`\\log_{10}M_0`.
        lM1_0 (float): offset parameter for
            :math:`\\log_{10}M_1`.
        lM1_p (float): tilt parameter for
            :math:`\\log_{10}M_1`.
        alpha_0 (float): offset parameter for
            :math:`\\alpha`.
        alpha_p (float): tilt parameter for
            :math:`\\alpha`.
        fc_0 (float): offset parameter for
            :math:`f_c`.
        fc_p (float): tilt parameter for
            :math:`f_c`.
        bg_0 (float): offset parameter for
            :math:`\\beta_g`.
        bg_p (float): tilt parameter for
            :math:`\\beta_g`.
        bmax_0 (float): offset parameter for
            :math:`\\beta_{\\rm max}`.
        bmax_p (float): tilt parameter for
            :math:`\\beta_{\\rm max}`.
        a_pivot (float): pivot scale factor :math:`a_*`.
        ns_independent (bool): drop requirement to only form
            satellites when centrals are present.
    """
    __repr_attrs__ = ("cM", "lMmin_0", "lMmin_p", "siglM_0", "siglM_p",
                      "lM0_0", "lM0_p", "lM1_0", "lM1_p", "alpha_0", "alpha_p",
                      "fc_0", "fc_p", "bg_0", "bg_p", "bmax_0", "bmax_p",
                      "a_pivot", "ns_independent", "precision_fftlog",)
    name = 'HOD'
    is_number_counts = True

    def __init__(self, c_M_relation,
                 lMmin_0=12., lMmin_p=0., siglM_0=0.4,
                 siglM_p=0., lM0_0=7., lM0_p=0.,
                 lM1_0=13.3, lM1_p=0., alpha_0=1.,
                 alpha_p=0., fc_0=1., fc_p=0.,
                 bg_0=1., bg_p=0., bmax_0=1., bmax_p=0.,
                 a_pivot=1., ns_independent=False):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`")

        self.cM = c_M_relation
        self.lMmin_0 = lMmin_0
        self.lMmin_p = lMmin_p
        self.lM0_0 = lM0_0
        self.lM0_p = lM0_p
        self.lM1_0 = lM1_0
        self.lM1_p = lM1_p
        self.siglM_0 = siglM_0
        self.siglM_p = siglM_p
        self.alpha_0 = alpha_0
        self.alpha_p = alpha_p
        self.fc_0 = fc_0
        self.fc_p = fc_p
        self.bg_0 = bg_0
        self.bg_p = bg_p
        self.bmax_0 = bmax_0
        self.bmax_p = bmax_p
        self.a_pivot = a_pivot
        self.ns_independent = ns_independent
        super(HaloProfileHOD, self).__init__()

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def update_parameters(self, lMmin_0=None, lMmin_p=None,
                          siglM_0=None, siglM_p=None,
                          lM0_0=None, lM0_p=None,
                          lM1_0=None, lM1_p=None,
                          alpha_0=None, alpha_p=None,
                          fc_0=None, fc_p=None,
                          bg_0=None, bg_p=None,
                          bmax_0=None, bmax_p=None,
                          a_pivot=None,
                          ns_independent=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to `None` won't be updated.

        Args:
            lMmin_0 (float): offset parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            lMmin_p (float): tilt parameter for
                :math:`\\log_{10}M_{\\rm min}`.
            siglM_0 (float): offset parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            siglM_p (float): tilt parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            lM0_0 (float): offset parameter for
                :math:`\\log_{10}M_0`.
            lM0_p (float): tilt parameter for
                :math:`\\log_{10}M_0`.
            lM1_0 (float): offset parameter for
                :math:`\\log_{10}M_1`.
            lM1_p (float): tilt parameter for
                :math:`\\log_{10}M_1`.
            alpha_0 (float): offset parameter for
                :math:`\\alpha`.
            alpha_p (float): tilt parameter for
                :math:`\\alpha`.
            fc_0 (float): offset parameter for
                :math:`f_c`.
            fc_p (float): tilt parameter for
                :math:`f_c`.
            bg_0 (float): offset parameter for
                :math:`\\beta_g`.
            bg_p (float): tilt parameter for
                :math:`\\beta_g`.
            bmax_0 (float): offset parameter for
                :math:`\\beta_{\\rm max}`.
            bmax_p (float): tilt parameter for
                :math:`\\beta_{\\rm max}`.
            a_pivot (float): pivot scale factor :math:`a_*`.
            ns_independent (bool): drop requirement to only form
                satellites when centrals are present
        """
        if lMmin_0 is not None:
            self.lMmin_0 = lMmin_0
        if lMmin_p is not None:
            self.lMmin_p = lMmin_p
        if lM0_0 is not None:
            self.lM0_0 = lM0_0
        if lM0_p is not None:
            self.lM0_p = lM0_p
        if lM1_0 is not None:
            self.lM1_0 = lM1_0
        if lM1_p is not None:
            self.lM1_p = lM1_p
        if siglM_0 is not None:
            self.siglM_0 = siglM_0
        if siglM_p is not None:
            self.siglM_p = siglM_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
        if fc_0 is not None:
            self.fc_0 = fc_0
        if fc_p is not None:
            self.fc_p = fc_p
        if bg_0 is not None:
            self.bg_0 = bg_0
        if bg_p is not None:
            self.bg_p = bg_p
        if bmax_0 is not None:
            self.bmax_0 = bmax_0
        if bmax_p is not None:
            self.bmax_p = bmax_p
        if a_pivot is not None:
            self.a_pivot = a_pivot
        if ns_independent is not None:
            self.ns_independent = ns_independent

    def _usat_real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = r_use[None, :] / (R_s[:, None] * bg)
        prof = 1./(x * (1 + x)**2)
        # Truncate
        prof[r_use[None, :] > R_M[:, None]*bmax] = 0

        norm = 1. / (4 * np.pi * (bg*R_s)**3 * (np.log(1+c_M) - c_M/(1+c_M)))
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _usat_fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = k_use[None, :] * R_s[:, None] * bg
        Si1, Ci1 = sici((1 + c_M[:, None]) * x)
        Si2, Ci2 = sici(x)

        P1 = 1. / (np.log(1+c_M) - c_M/(1+c_M))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
        prof = P1[:, None] * (P2 - P3)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        ur = self._usat_real(cosmo, r_use, M_use, a, mass_def)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * ur
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * ur)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a, mass_def)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * uk
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a, mass_def)

        prof = Ns[:, None] * uk
        if self.ns_independent:
            prof = 2 * Nc[:, None] * fc * prof + prof**2
        else:
            prof = Nc[:, None] * (2 * fc * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fc(self, a):
        # Observed fraction of centrals
        return self.fc_0 + self.fc_p * (a - self.a_pivot)

    def _Nc(self, M, a):
        # Number of centrals
        Mmin = 10.**(self.lMmin_0 + self.lMmin_p * (a - self.a_pivot))
        siglM = self.siglM_0 + self.siglM_p * (a - self.a_pivot)
        return 0.5 * (1 + erf(np.log(M/Mmin)/siglM))

    def _Ns(self, M, a):
        # Number of satellites
        M0 = 10.**(self.lM0_0 + self.lM0_p * (a - self.a_pivot))
        M1 = 10.**(self.lM1_0 + self.lM1_p * (a - self.a_pivot))
        alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
        return np.heaviside(M-M0, 1) * (np.fabs(M-M0) / M1)**alpha
