import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.special
import scipy.integrate
from .. import ccllib as lib
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power
from ..background import growth_factor

try:
    import fastpt as fpt
    HAVE_FASTPT = True
except ImportError:
    HAVE_FASTPT = False

a0 = 1.

class PTNLEffCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    the non-linear power spectrum used in the effective halo model by Philcox et al., 2020.
    These calculations are currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        cs2 (float): Squared-speed-of-sound counterterm :math:`c_s^2` in :math:`(\mathrm{Mpc})^2` units.
        R: Smoothing scale in :math:`\mathrm{Mpc}` units.
    Keyword Args:
        pade_resum (bool): If True, use a Pade resummation of the counterterm
        :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
        smooth_density (bool): If True, smooth the density field on scale R, i.e.
        multiply power by W(kR)^2, default: True
    """

    def __init__(self, cosmo, cs2, R, pade_resum=True, smooth_density=True, z_evol=False):

        self.cosmo = cosmo
        self.z_evol = z_evol
        self.cs2 = cs2
        self.R = R
        self.pade_resum = pade_resum
        self.smooth_density = smooth_density

        # Set other hyperparameters consistently. (These are non-critical but control minutae of IR resummation and
        # interpolation precision)
        self.IR_N_k = 5000
        self.IR_k_max = 1.*self.cosmo.cosmo.params.h
        self.OneLoop_N_interpolate = 30
        self.OneLoop_k_cut = 3
        self.OneLoop_N_k = 2000

    def compute_one_loop_only_power(self, k, pk_lin_interp):
        """
        Compute the one-loop SPT power from the linear power spectrum in the Cosmology class.
        This returns the one-loop power evaluated at the wavenumber vector specfied in the function call.
        Args:
            k (np.ndarray): an array holding values of the wave vector in Mpc^-1 units
            at which the power spectrum should be calculated.
            pk_lin_interp (function): Function taking input wavenumber in Mpc^-1 units and returning a linear
            power spectrum.
        Returns:
            np.ndarray: Vector of 1-loop power :math:`P_\mathrm{1-loop}(k)` for the input k-vector.
        """

        one_loop_only_power = self._one_loop_only_power(k, pk_lin_interp)

        return one_loop_only_power

    def compute_resummed_one_loop_power(self, k, a, ga4, ga2, pk_lin_interp):
        """
        Compute the IR-resummed linear-plus-one-loop power spectrum, using the linear power spectrum in the
        Cosmology class.
        The output power is defined by
        .. math::
            P_\mathrm{lin+1, IR}(k) = P_\mathrm{lin, nw}(k) + P_\mathrm{1-loop, nw}(k) +
            e^{-k^2\Sigma^2} [ P_\mathrm{lin, w}(k) (1 + k^2\Sigma^2) + P_\mathrm{1-loop,w}(k) ]
        where 'nw' and 'w' refer to the no-wiggle and wiggle parts of the linear / 1-loop power spectrum and
        :math:`Sigma^2` is the BAO damping scale (computed in the prepare_IR_resummation function)
        Args:
            k (np.ndarray): an array holding values of the wave vector in Mpc^-1 units
            at which the power spectrum should be calculated.
            a (np.ndarray): an array holding values of the scale factor
            at which the power spectrum should be calculated.
            ga4 (np.ndarray): an array holding values of the growth factor to the power of four :math:`D(a)^4`
            for scale factor values in :math:`a`.
            ga2 (np.ndarray): an array holding values of the growth factor squared :math:`D(a)^2`
            for scale factor values in :math:`a`.
            pk_lin_interp (function): Function taking input wavenumber in Mpc^-1 units and returning a linear
            power spectrum.
        Returns:
            np.ndarray: Vector of IR-resummed linear-plus-one-loop power :math:`P_\mathrm{lin+1,IR}(k)`
            for the input k-vector.
        """

        if not hasattr(self, 'linear_no_wiggle_power'):
            # First create IR interpolators if not present
            self.prepare_IR_resummation(k, a)

        # Compute 1-loop only power spectrum
        one_loop_all = self.compute_one_loop_only_power(k, pk_lin_interp)

        # Load no-wiggle and wiggly parts
        no_wiggle_lin = self.linear_no_wiggle_power
        wiggle_lin = self.linear_power - no_wiggle_lin
        no_wiggle_one_loop = self.one_loop_only_no_wiggle_power
        wiggle_one_loop = one_loop_all - no_wiggle_one_loop

        # Compute and return IR resummed power
        resummed_one_loop_power = ga2*no_wiggle_lin + ga4*no_wiggle_one_loop + np.exp(
            -self.BAO_damping[:, np.newaxis] * k ** 2.) * (ga2*wiggle_lin * (
                    1. + k ** 2. * self.BAO_damping[:, np.newaxis]) + ga4*wiggle_one_loop)

        return resummed_one_loop_power

    def prepare_IR_resummation(self, k, a):
        """
        Compute relevant quantities to allow IR resummation of the non-linear power spectrum to be performed.
        This computes the no-wiggle power spectrum, from the 4th order polynomial scheme of Hamann et al. 2010.
        A group of spectra for the no-wiggle linear and no-wiggle 1-loop power are output for later use. The
        BAO damping scale
        .. math::
            \Sigma^2 =  \frac{1}{6\pi^2}\int_0^\Lambda dq\,P_\mathrm{NL}^{nw}(q)\left[1-j_0(q\ell_\mathrm{BAO})+
            2j_2(q\ell_\mathrm{BAO})\right]
        is also computed.
        Args:
            k (np.ndarray): an array holding values of the wave vector
            at which the power spectrum should be calculated.
            a (np.ndarray): an array holding values of the scale factor
            at which the power spectrum should be calculated.
        Reurns:
        """

        # First define a k-grid in Mpc^-1 units
        min_k = np.max([np.min(k),1e-4]) # setting minimum to avoid zero errors
        max_k = np.max(k)
        k_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,self.IR_N_k)

        # Define turning point of power spectrum (we compute no-wiggle spectrum beyond this point)
        linear_power_interp = linear_matter_power(self.cosmo, k_interp, a0)
        # Create interpolant for linear power spectrum
        self.linear_power_interp = scipy.interpolate.interp1d(k_interp, linear_power_interp)
        max_pos = np.where(linear_power_interp==max(linear_power_interp))
        k_turn = k_interp[max_pos]
        Pk_turn = linear_power_interp[max_pos]
        Pk_max = linear_matter_power(self.cosmo, np.atleast_1d(self.IR_k_max), a0)

        # Define k in required range
        ffilt = np.where(np.logical_and(k_interp>k_turn,k_interp<self.IR_k_max))
        k_filt = k_interp[ffilt]

        # Compute ln(P(k)) in region
        log_Pk_mid = np.log(linear_power_interp[ffilt])
        logP1 = np.log(Pk_turn)
        logP2 = np.log(Pk_max)

        # Now fit a fourth order polynomial to the data, fixing the values at the edges.
        def _fourth_order_poly(k,coeff):
            a2,a3,a4=coeff
            poly24 = lambda lk: a2*lk**2.+a3*lk**3.+a4*lk**4.
            f1 = logP1 - poly24(np.log(k_turn))
            f2 = logP2 - poly24(np.log(self.IR_k_max))
            a1 = (f1-f2)/(np.log(k_turn)-np.log(self.IR_k_max))
            a0 = f1 - a1*np.log(k_turn)
            return a0+a1*np.log(k)+poly24(np.log(k))

        def _fourth_order_fit(coeff):
            return ((log_Pk_mid-_fourth_order_poly(k_interp[ffilt],coeff))**2.).sum()

        poly_fit = scipy.optimize.minimize(_fourth_order_fit, [0.,0.,0.])

        # Compute the no-wiggle spectrum, inserting the smooth polynomial in the required range
        noWiggleSpec = linear_power_interp
        noWiggleSpec[ffilt] = np.exp(_fourth_order_poly(k_filt,poly_fit.x))

        # Now compute no-wiggle power via interpolater
        linear_no_wiggle_power_interp = scipy.interpolate.interp1d(k_interp,noWiggleSpec)
        self.linear_no_wiggle_power = linear_no_wiggle_power_interp(k)
        self.linear_power = linear_matter_power(self.cosmo, k, a0)

        # Compute one-loop interpolator for no-wiggle power
        # This is just the one-loop operator acting on the no-wiggle power spectrum
        self.one_loop_only_no_wiggle_power = self._one_loop_only_power(k, linear_no_wiggle_power_interp)

        # Compute the BAO smoothing scale Sigma^2
        def _BAO_integrand(q):
            r_BAO = 105./self.cosmo.cosmo.params.h # BAO scale in Mpc
            kh_osc = 1./r_BAO
            pk_lin = np.array([linear_matter_power(self.cosmo, q, ai) for ai in a])
            return pk_lin*(1. - scipy.special.spherical_jn(0, q/kh_osc) +
                   2.*scipy.special.spherical_jn(2, q/kh_osc))/(6.*np.pi**2.)

        kk_grid = np.linspace(1e-4,0.2,10000)

        # Now store the BAO damping scale as Sigma^2
        self.BAO_damping = scipy.integrate.simps(_BAO_integrand(kk_grid),kk_grid, axis=1)

    def _one_loop_only_power(self, k, pk_lin_interp):
        """
        Compute the one-loop SPT power spectrum, using the FAST-PT module. This is computed from an input
        linear power spectrum.
        Note that the FAST-PT output contains large oscillations at high-k. To alleviate this, we perform
        smoothing interpolation above some k.
        Args:
            pk_lin_interp (function): Function taking input wavenumber in Mpc^-1 units and returning a linear
            power spectrum.
        Returns:
            np.ndarray: The one-loop SPT power spectrum given an input k (in :math:`h/\mathrm{Mpc}` units).
        """

        # First define a k-grid in Mpc^-1 units
        min_k = np.max([np.min(k),1e-4]) # setting minimum to avoid zero errors
        max_k = np.max(k)
        k_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,self.IR_N_k)

        # Compute the one-loop spectrum using FAST-PT
        fpt_obj = fpt.FASTPT(k_interp, to_do=['one_loop_dd'], n_pad=len(k)*3, verbose=0)
        initial_power = fpt_obj.one_loop_dd(pk_lin_interp(k_interp), C_window=0.65, P_window=[0.25, 0.25])[0]

        # Now convolve k if necessary
        filt = k_interp > self.OneLoop_k_cut
        if np.sum(filt)==0:
            combined_power = initial_power
            combined_k = k_interp
        else:
            convolved_k = np.convolve(k_interp[filt], np.ones(self.OneLoop_N_interpolate,)/self.OneLoop_N_interpolate,
                                      mode='valid')
            convolved_power = np.convolve(initial_power[filt],
                                    np.ones(self.OneLoop_N_interpolate,)/self.OneLoop_N_interpolate, mode='valid')

            # Concatenate to get an output
            combined_power = np.concatenate([initial_power[k_interp<min(convolved_k)],convolved_power])
            combined_k = np.concatenate([k_interp[k_interp<min(convolved_k)],convolved_k])

        pk_1l_int = scipy.interpolate.interp1d(combined_k, combined_power)
        combined_power = pk_1l_int(k)

        return combined_power

    def compute_smoothing_function(self, k, R):
        """
        Compute the smoothing function :math:`W(kR)`, for smoothing scale R. This accounts for the smoothing of
        the density field on scale R and is the Fourier transform of a spherical top-hat of scale R.
        Args:
            R: Smoothing scale in :math:`\mathrm{Mpc}` units.
        Returns:
            np.ndarray: :math:`W(kR)` evaluated on the input k-vector.
        """

        kR = k*R
        w = 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3.

        return w

def get_pt_eff_pk2d(cosmo, ptc):
    """
    Compute the non-linear power spectrum to one-loop order, with IR corrections and counterterms.
    Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean
    parameters pade_resum, smooth_density=False.
    Including all relevant effects, this is defined as
    .. math::
        P_\mathrm{NL}(k, R, c_s^2) = [P_\mathrm{lin}(k) + P_\mathrm{1-loop}(k) +
        P_\mathrm{counterterm}(k;c_s^2)] W(kR)
    where
    .. math::
        P_\mathrm{counterterm}(k;c_s^2) = - c_s^2 \\frac{k^2 }{(1 + k^2)} P_\mathrm{lin}(k)
    is the counterterm, and IR resummation is applied to all spectra.
    This computes the relevant loop integrals if they haven't already been computed. The function returns
    a :class:`~pyccl.pk2d.Pk2D` object containing the PT power spectrum given smoothing scale R and
    effective squared sound-speed :math:`c_s^2`.
    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        ptc (:class:`PTNLEffCalculator`): a perturbation theory calculator.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
    """

    status = 0
    # Set k and a sampling from CCL parameters
    nk = lib.get_pk_spline_nk(cosmo.cosmo)
    na = lib.get_pk_spline_na(cosmo.cosmo)
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status)
    lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
    check(status)
    k = np.exp(lk_arr)

    # Linear growth factor
    ga = growth_factor(cosmo, a_arr)
    ga4 = ga[:, np.newaxis]**4
    ga2 = ga[:, np.newaxis]**2

    # Redshift-evolution of effective halo model parameters from Oliver Philcox (from fit to Quijote sims)
    if ptc.z_evol:
        cs2 = ptc.cs2*ga[:, np.newaxis]**3
        R = ptc.R*ga2
    else:
        cs2 = ptc.cs2
        R = ptc.R

    ptc.prepare_IR_resummation(k, a_arr)
    counterterm_tmp = -cs2 * k ** 2.
    if ptc.pade_resum:
        counterterm_tmp /= (1. + (k/ptc.cosmo.cosmo.params.h) ** 2.)
    no_wiggle_lin = ptc.linear_no_wiggle_power
    wiggle_lin = ptc.linear_power - no_wiggle_lin
    p_pt = ptc.compute_resummed_one_loop_power(k, a_arr, ga4, ga2, ptc.linear_power_interp) + ga2*counterterm_tmp * (
                no_wiggle_lin + wiggle_lin * np.exp(-ptc.BAO_damping[:, np.newaxis] * k ** 2.))

    if ptc.smooth_density:
        p_pt *= ptc.compute_smoothing_function(k, R) ** 2.

    pt_pk = Pk2D(a_arr=a_arr,
                 lk_arr=np.log(k),
                 pk_arr=p_pt,
                 is_logp=False)

    # Now account for neutrino effects if present
    # if ptc.cosmology.use_neutrinos:
    #     if ptc.include_neutrinos:
    #         f_nu = self.cosmo.Omega_nu / self.cosmo.Omega_m()
    #         f_cb = 1. - ptc.cosmology.f_nu
    #         return f_cb ** 2. * output + (ptc.linear_power_total - f_cb ** 2. * ptc.linear_power)
    #     else:
    #         return output

    return pt_pk

