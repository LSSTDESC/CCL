from .. import ccllib as lib
#from ..core import check
from ..background import angular_diameter_distance
from .profiles import HaloProfile
from .massdef import MassDef
import numpy as np


class WLCalculator(object):
    """ This class enables the calculation of weak lensing functions:
    convergence, shear, reduced shear and magnification.
    These functions depends on the halo profile :math:`\\rho(R)` and the 
    critical surface mass density :math:`\\Sigma_{\\mathrm{crit}}`. The 
    last reads:
    .. math::
        \\Sigma_{\\mathrm{crit}} = \\frac{c^2}{4\\pi G} 
        \\frac{D_{\\rm{s}}}{D_{\\rm{l}}D_{\\rm{ls}}}, 
    
    where :math:`c` is the speed of light, :math:`G` is the gravitational 
    constant, and :math:`D_i` is the angular diameter distance. The labels 
    :math:`i =` s, l and ls denotes the distances to the source, lens, and 
    between source and lens, respectively.  
    
    Convergence:
    .. math::
        \\kappa(R) = \\frac{\\Sigma_R}{\\Sigma_{\\mathrm{crit}}}\\,
        
    where :math:`\\Sigma(R)` is the 2D projected surface mass density.

    Shear:
    .. math::
        \\gamma(R) = 
        
    Reduced shear:
    .. math::
        g (R) = 
        
    Magnification:
    .. math::    
        \\mu = 
        
    Args:
        cosmo    (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        hprofile (:class:`pyccl.halos.profile.HaloProfile`): A HaloProfile object
    """
    name = 'default'

    def __init__(self, cosmo, hprofile):
        # Check if cosmo and halo profile were provided.
        if cosmo is None:
            raise ValueError("Cosmology was not initialized.")
        else:
            self.cosmo = cosmo
        if hprofile is None:
            raise ValueError("Halo profile was not initialized.")
        else:
            self.hprofile = hprofile

    def critical_sigma (self, cosmo, alens, asource):
        """ Returns \\Sigma_{\\mathrm{crit}}.
        
        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            alens (float): lens' scale factor.
            asource (float or array_like): source's scale factor.
            
        Returns:
            float or array_like: :math:`\\Sigma_{\\mathrm{crit}}` in units of :math:`\\M_{\\odot}/Mpc^2`    
        """
        Ds  = angular_diameter_distance(cosmo, asource, a2=None) 
        Dl  = angular_diameter_distance(cosmo, alens, a2=None)
        Dls = angular_diameter_distance(cosmo, alens, asource)
        A   = ccl_constants.CLIGHT**2 * ccl_constants.MPC_TO_METER / 
              (4.0 * np.pi * ccl_constants.GNEWT * ccl_constants.SOLAR_MASS)
              
        Sigma_crit = A * Ds / (Dl * Dls)
              
        return Sigma_crit  

    def convergence (self, cosmo, hprofile, r, alens, M, mdef):
        """ Returns the convergence for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            hprofile (:class:`~pyccl.halos.profile.HaloProfile): A HaloProfile object.
            r (float or array_like): comoving radius in Mpc.
            alens (float): lens' scale factor.
            M (float or array_like): halo mass in units of M_sun.
            mdef (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: convergence \
                :math:`\\kappa`
        """
        
        Sigma = hprofile.projected (cosmo, r, M, alens, mdef)
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status)
        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, logM,
                                                   len(logM), status)
        check(status)

        rho = (lib.cvar.constants.RHO_CRITICAL *
               cosmo['Omega_m'] * cosmo['h']**2)
        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use

        if np.ndim(M) == 0:
            mf = mf[0]
        return mf

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        """ Get the :math:`f(\\sigma_M)` function for this mass function
        object (see description of this class for details).

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            sigM (float or array_like): standard deviation in the
                overdensity field on the scale of this halo.
            a (float): scale factor.
            lnM (float or array_like): natural logarithm of the
                halo mass in units of M_sun (provided in addition
                to sigM for convenience in some mass function
                parametrizations).

        Returns:
            float or array_like: :math:`f(\\sigma_M)` function.
        """
        raise NotImplementedError("Use one of the non-default "
                                  "MassFunction classes")


