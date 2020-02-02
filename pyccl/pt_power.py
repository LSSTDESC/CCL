import numpy as np
from scipy.interpolate import interp1d
from .pyutils import _check_array_params
from .pk2d import Pk2D
from .power import linear_matter_power, nonlin_matter_power
from .background import growth_factor

try:
    import fastpt as fpt
    HAVE_FASTPT = True
except ImportError:
    HAVE_FASTPT = False


class PTTracer(object):
    def __init_(self):
        self.biases = {}
        self.type = None
        pass

    def get_bias(self, bias_name, z):
        if bias_name not in self.biases:
            raise KeyError("Bias %s not included in this tracer" % bias_name)
        return self.biases[bias_name](z)

    def _get_bias_function(self, b):
        # If None, assume it's zero
        if b is None:
            b = 0

        # If it's a scalar, then assume it's a constant function
        if np.ndim(b) == 0:
            def _const(z):
                if np.ndim(z) == 0:
                    return b
                else:
                    return b * np.ones_like(z)

            return _const
        else:  # Otherwise interpolate
            z, b = _check_array_params(b)
            return interp1d(z, b, bounds_error=False,
                            fill_value=b[-1])


class PTNumberCountsTracer(PTTracer):
    def __init__(self, b1, b2=None, bs=None):
        self.biases = {}
        self.type = 'NC'

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2
        self.biases['b2'] = self._get_bias_function(b2)
        # Initialize bs
        self.biases['bs'] = self._get_bias_function(bs)

    @property
    def b1(self):
        return self.biases['b1']

    @property
    def b2(self):
        return self.biases['b2']

    @property
    def bs(self):
        return self.biases['bs']


class PTIntrinsicAlignmentTracer(PTTracer):
    def __init__(self, c1, c2=None, cdelta=None):

        self.biases = {}
        self.type = 'IA'

        # Initialize c1
        self.biases['c1'] = self._get_bias_function(c1)
        # Initialize c2
        self.biases['c2'] = self._get_bias_function(c2)
        # Initialize cdelta
        self.biases['cdelta'] = self._get_bias_function(cdelta)

    @property
    def c1(self):
        return self.biases['c1']

    @property
    def c2(self):
        return self.biases['c2']

    @property
    def cdelta(self):
        return self.biases['cdelta']


class PTWorkspace(object):
    def __init__(self, with_NC=True, with_IA=True,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 pad_factor=1, low_extrap=-5, high_extrap=3):
        self.with_NC = with_NC
        self.with_IA = with_IA

        to_do = []
        if self.with_IA:
            to_do.append('IA')
        if self.with_NC:
            to_do.append('dd_bias')

        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        n_pad = pad_factor * len(self.ks)

        self.pt_object = fpt.FASTPT(self.ks, to_do=to_do,
                                    low_extrap=low_extrap,
                                    high_extrap=high_extrap,
                                    n_pad=n_pad)


def get_pt_pk(cosmo, k, a, tracer_1, tracer_2=None,
              pt_object=None, return_pt_object=False,
              sub_lowk=True, use_nonlin=True):
    # First initialize the pt_object if needed
    if pt_object is not None:
        # we will need to check that the input k was the
        # same used to generate the input pt_object
        ks = k
    if pt_object is None:
        assert HAVE_FASTPT, (
            "You must have the `fast-pt` python package "
            "installed to run CCL with FASTPT!")
        to_do = []
        if tracer_2 is None:
            tracer_2 = tracer_1
        if (tracer_1.type == 'NC') or (tracer_2.type == 'NC'):
            to_do.append('dd_bias')
        if (tracer_1.type == 'IA') or (tracer_2.type == 'IA'):
            to_do.append('IA')

        # other functionality (e.g. RSD) can be added here for FASTPT.
        # Ideally, the initialization happens once per likelihood
        # evaluation, or even once per chain.
        # Hard code some accuracy settings. These should be passable
        # at some point.
        pad_factor = 1
        low_extrap = -5
        high_extrap = 3
        P_window = None
        C_window = .75

        # we need a k array and the linear power spectrum at z=0.
        # could eventually run this at any z and avoid requiring
        # growth factor multiplication. Save this for later dev.

        # is the input k already logspaced?
        # is it sufficiently high resoluton for the calculation
        dk = np.diff(np.log(k))
        delta_L = (np.log(k[-1]) - np.log(k[0])) / (k.size - 1)
        dk_test = np.ones_like(dk) * delta_L
        log_sample_test = 'FASTPT will not work if your input (k,Pk)'
        log_sample_test += 'values are not sampled evenly in log space. '
        log_sample_test += 'Creating a new array of k values now.'
        try:
            np.testing.assert_array_almost_equal(dk, dk_test, decimal=4,
                                                 err_msg=log_sample_test,
                                                 verbose=False)
            ks = k
        except AssertionError:
            # create a new k array
            # should eventually check that the range is enough, etc.
            ks = np.logspace(np.log10(k[0]),
                             np.log10(k[-1]),
                             (np.log10(k[-1]) - np.log10(k[0])) * 20)
        pk_lin_z0 = linear_matter_power(cosmo, ks, 1)

        # actually do the initialization
        n_pad = pad_factor*len(ks)
        pt_object = fpt.FASTPT(ks, to_do=to_do,
                               low_extrap=low_extrap,
                               high_extrap=high_extrap,
                               n_pad=n_pad)

    # Now compute the Pk using FASTPT
    # First, get P_d1d1 (the delta delta correlation), which could
    # be linear or nonlinear.
    # We will eventually want options here, e.g. to use pert theory
    # instead of halofit.
    if use_nonlin:
        Pd1d1 = np.array([nonlin_matter_power(cosmo, ks, a_i) for a_i in a])
    else:
        Pd1d1 = np.array([linear_matter_power(cosmo, ks, a_i) for a_i in a])
    # We also need the growth factor
    ga = growth_factor(cosmo, a)
    # NOTE: we should add an option that does PT at every a value.
    # This isn't hard, it just takes longer.

    if (tracer_1.type == 'NC') and (tracer_2.type == 'NC'):
        # note that the bias is returned as a function of z.
        # To get a single value, you need to put in a z value
        # is it looking for z, or scale factor?
        b1_1 = tracer_1.b1(0)
        b1_2 = tracer_2.b1(0)
        b2_1 = tracer_1.b2(0)
        b2_2 = tracer_2.b2(0)
        bs_1 = tracer_1.bs(0)
        bs_2 = tracer_2.bs(0)

        bias_fpt = pt_object.one_loop_dd_bias(pk_lin_z0,
                                              P_window=P_window,
                                              C_window=C_window)

        # replace with np.outer?
        Pd1d2 = np.array([g**4 * bias_fpt[2] for g in ga])
        Pd2d2 = np.array([g**4 * bias_fpt[3] for g in ga])
        Pd1s2 = np.array([g**4 * bias_fpt[4] for g in ga])
        Pd2s2 = np.array([g**4 * bias_fpt[5] for g in ga])
        Ps2s2 = np.array([g**4 * bias_fpt[6] for g in ga])
        sig4ka = 0.
        if sub_lowk:
            sig4ka = np.array([g**4 * bias_fpt[7] *
                               np.ones_like(bias_fpt[0]) for g in ga])
        p_gg = (b1_1*b1_2*Pd1d1 + (1./2)*(b1_1*b2_2+b1_2*b2_1)*Pd1d2 +
                (1./4)*b2_1*b2_2*(Pd2d2-2.*sig4ka) +
                (1./2)*(b1_1*bs_2+b1_2*bs_1)*Pd1s2 +
                (1./4)*(b2_1*bs_2+b2_2*bs_1)*(Pd2s2-4./3*sig4ka) +
                (1./4)*bs_1*bs_2*(Ps2s2-8./9*sig4ka))
        p_pt = p_gg

    elif (tracer_1.type == 'NC') and (tracer_2.type == 'IA'):
        p_pt = Pd1d1  # placeholder
    elif (tracer_1.type == 'IA') and (tracer_2.type == 'NC'):
        p_pt = Pd1d1  # placeholder
    elif (tracer_1.type == 'IA') and (tracer_2.type == 'NC'):
        p_pt = Pd1d1  # placeholder
    else:
        raise ValueError('Combination of %s and %s types not supported.' %
                         (tracer_1.type, tracer_2.type))

    # I've initialized the tracers to None, assuming that,
    # if tracer_2 is None, then the assumption is that
    # it is the same as tracer_1.

    # for matter cross correlation, we will use (for now) a
    # PTNumberCounts tracer with b1=1 and all other bias = 0

    # at some point, do we want to store an spline array for these
    # pt power?

    # nonlinear bias cross-terms with IA are currently missing from
    # here and from FASTPT come back to this.

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = Pk2D(a_arr=a,
                 lk_arr=np.log(ks),
                 pk_arr=p_pt,
                 is_logp=False)  # ?
    # (see pk2d.py for other options).

    if return_pt_object:
        return pt_pk, pt_object
    else:
        return pt_pk
