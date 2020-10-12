from .. import ccllib as lib
from ..background import angular_diameter_distance
from .profiles import HaloProfile
from .massdef import MassDef
import numpy as np

physical_constants = lib.cvar.constants

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
        \\kappa(R) = \\frac{\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},\\
        
    where :math:`\\Sigma(R)` is the 2D projected surface mass density.

    Shear (tangential):
    .. math::
        \\gamma(R) = \\frac{\\Delta\\Sigma(R)}{\\Sigma_{\\mathrm{crit}}} = \\frac{\\overline{\\Sigma}(< R) - \\Sigma(R)}{\\Sigma_{\\mathrm{crit}}},\\
        
    where :math:`\\overline{\\Sigma}(< R)` is the average surface density within R.     
        
    Reduced shear:
    .. math::
        g (R) = \\frac{\\gamma(R)}{(1 - \\kappa(R))}.\\
        
    Magnification:
    .. math::    
        \\mu (R) = \\frac{1}{\\left[(1 - \\kappa(R))^2 - \\vert \\gamma(R) \\vert^2 \\right]]}.\\
        
    Args:
        cosmo    (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        hprofile (:class:`~pyccl.halos.profile.HaloProfile`): A HaloProfile object.
        massdef  (:class:`~pyccl.halos.massdef.MassDef`): a mass definition object.
        M: mass in units of M_sun.
    """
    name = 'default'

    def __init__(self, cosmo, hprofile, massdef):
        # Check if cosmo and halo profile were provided.
        if cosmo is None:
            raise ValueError("Cosmology was not initialized.")
        else:
            self.cosmo = cosmo
        if hprofile is None:
            raise ValueError("Halo profile was not initialized.")
        else:
            self.hprofile = hprofile
        if massdef is None:
            raise ValueError("Mass definition was not initialized.")
        else:
            self.mdef = massdef        
            
    def sigma_critical (self, a_lens, a_src):
        """ Returns \\Sigma_{\\mathrm{crit}}.
        
        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            a_lens (float): lens' scale factor.
            a_src (float or array_like): source's scale factor.
            
        Returns:
            float or array_like: :math:`\\Sigma_{\\mathrm{crit}}` in units of :math:`\\M_{\\odot}/Mpc^2`    
        """
        Ds  = angular_diameter_distance(self.cosmo, a_src, a2=None) 
        Dl  = angular_diameter_distance(self.cosmo, a_lens, a2=None)
        Dls = angular_diameter_distance(self.cosmo, a_lens, a_src)
        A   = physical_constants.CLIGHT**2 * physical_constants.MPC_TO_METER / (4.0 * np.pi * physical_constants.GNEWT * physical_constants.SOLAR_MASS)
              
        Sigma_crit = A * Ds / (Dl * Dls)
              
        return Sigma_crit  

    def convergence (self, r, M, a_lens, a_src):
        """ Returns the convergence for input parameters.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: convergence \
                :math:`\\kappa`
        """
        
        Sigma = self.hprofile.projected (self.cosmo, r, M, a_lens, self.mdef)
        Sigma_crit = self.sigma_critical (a_lens, a_src) 
        return Sigma / Sigma_crit

    def shear (self, r, M, a_lens, a_src):
        """ Returns the shear for input parameters.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: shear \
                :math:`\\gamma`
        """
        
        Sigma      = self.hprofile.projected (self.cosmo, r, M, a_lens, self.mdef)
        Sigma_bar  = self.hprofile.cumul2d (self.cosmo, r, M, a_lens, self.mdef)
        Sigma_crit = self.sigma_critical (a_lens, a_src) 
        return (Sigma_bar - Sigma) / Sigma_crit

    def reduced_shear (self, r, M, a_lens, a_src):
        """ Returns the reduced shear for input parameters.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: reduced shear \
                :math:`g`
        """
        
        convergence = self.convergence (r, M, a_lens, a_src)
        shear       = self.shear (r, M, a_lens, s_src)
        return shear / (1.0 - convergence)
        
    def reduced_shear_physical (self, r, M, a_lens, a_src):
        """ Returns the reduced shear for input parameters in physical units.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: reduced shear in physical units (not comoving)\
                :math:`g`
        """
        
        convergence = self.convergence (r, M, a_lens, a_src)
        shear       = self.shear (r, M, a_lens, a_src)
        return shear / (a_lens**2 - convergence)        

    def magnification (self, r, M, a_lens, a_src):
        """ Returns the magnification for input parameters.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: magnification\
                :math:`\\mu`    
        """
    
        convergence = self.convergence (r, M, a_lens, a_src)
        shear       = self.shear (r, M, a_lens, a_src)

        return 1.0 / ((1.0 - convergence)**2 - np.abs (shear)**2)
             
    def magnification_physical (self, r, M, a_lens, a_src):
        """ Returns the magnification for input parameters in physical units.

        Args:
            r (float or array_like): comoving radius in Mpc.
            M (float or array_like): halo mass in umits of M_sun.
            a_lens (float or array_like): lens' scale factor.
            a_src (float or array_like): source's scale factor.

        Returns:
            float or array_like: magnification in physical units (not comoving)\
                :math:`\\mu`    
        """
    
        al2       = a_lens**2
        conv_phy  = self.convergence (r, M, a_lens, a_src) / al2
        shear_phy = self.shear (r, M, a_lens, a_src) / al2

        return 1.0 / ((1.0 - conv_phy)**2 - np.abs (shear_phy)**2)
          