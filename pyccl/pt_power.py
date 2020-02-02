import numpy as np
from scipy.interpolate import interp1d
from . import ccllib as lib
from .pyutils import _check_array_params
from .core import check
from .pk2d import Pk2D
from .power import linear_matter_power, nonlin_matter_power
from .background import growth_factor

try:
    import fastpt as fpt
    HAVE_FASTPT = True
except ImportError:
    HAVE_FASTPT = False


class PTTracer(object):
    def __init__(self):
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


class PTMatterTracer(PTTracer):
    def __init__(self):
        self.biases = {}
        self.type = 'M'


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
                 pad_factor=1, low_extrap=-5, high_extrap=3,
                 P_window=None, C_window=.75):
        self.with_NC = with_NC
        self.with_IA = with_IA
        # TODO: what is this?
        self.P_window = P_window
        self.C_window = C_window

        to_do = ['one_loop_dd']
        if self.with_NC:
            to_do.append('dd_bias')
        if self.with_IA:
            to_do.append('IA')

        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        n_pad = pad_factor * len(self.ks)

        self.pt_object = fpt.FASTPT(self.ks, to_do=to_do,
                                    low_extrap=low_extrap,
                                    high_extrap=high_extrap,
                                    n_pad=n_pad)

    def get_dd_bias(self, pk):
        return self.pt_object.one_loop_dd_bias(pk,
                                               P_window=self.P_window,
                                               C_window=self.C_window)


def get_pt_pk2d(cosmo, w, tracer1, tracer2=None,
                sub_lowk=False, use_nonlin=True, a_arr=None,
                extrap_order_lok=1, extrap_order_hik=2):

    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status)

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, PTTracer):
        raise TypeError("tracer1 must be of type `PTTracer`")
    if not isinstance(tracer2, PTTracer):
        raise TypeError("tracer2 must be of type `PTTracer`")

    if (tracer1.type == 'NC') or (tracer2.type == 'NC'):
        if not w.with_NC:
            raise ValueError("Need number counts bias, "
                             "but workspace didn't compute it")
    if (tracer1.type == 'IA') or (tracer2.type == 'IA'):
        if not w.with_IA:
            raise ValueError("Need intrinsic alignment bias, "
                             "but workspace didn't compute it")

    # z
    z_arr = 1. / a_arr - 1
    # P_lin(k) at z=0
    pk_lin_z0 = linear_matter_power(cosmo, w.ks, 1.)
    # Linear growth factor
    ga = growth_factor(cosmo, a_arr)
    ga2 = ga**2
    ga4 = ga2**2
    # NOTE: we should add an option that does PT at every a value.
    # This isn't hard, it just takes longer.

    # Now compute the Pk using FASTPT
    # First, get P_d1d1 (the delta delta correlation), which could
    # be linear or nonlinear.
    # We will eventually want options here, e.g. to use pert theory
    # instead of halofit.
    if use_nonlin:
        Pd1d1 = np.array([nonlin_matter_power(cosmo, w.ks, a)
                          for a in a_arr]).T
    else:
        Pd1d1 = np.array([linear_matter_power(cosmo, w.ks, a)
                          for a in a_arr]).T

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.b2(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.b2(z_arr)

            bias_fpt = w.get_dd_bias(pk_lin_z0)

            Pd1d2 = ga4[None, :] * bias_fpt[2][:, None]
            Pd2d2 = ga4[None, :] * bias_fpt[3][:, None]
            Pd1s2 = ga4[None, :] * bias_fpt[4][:, None]
            Pd2s2 = ga4[None, :] * bias_fpt[5][:, None]
            Ps2s2 = ga4[None, :] * bias_fpt[6][:, None]
            s4 = 0.
            if sub_lowk:
                s4 = ga4 * bias_fpt[7]
                s4 = s4[None, :]

            p_gg = ((b11*b12)[None, :] * Pd1d1 +
                    0.5*(b11*b22 + b12*b21)[None, :] * Pd1d2 +
                    0.25*(b21*b22)[None, :] * (Pd2d2 - 2.*s4) +
                    0.5*(b11*b22 + b12*bs1)[None, :] * Pd1s2 +
                    0.25*(b21*bs2 + b22*bs1)[None, :] * (Pd2s2 - (4./3.)*s4) +
                    0.25*(bs1*bs2)[None, :] * (Ps2s2 - (8./9.)*s4))
            p_pt = p_gg
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # I've initialized the tracers to None, assuming that,
    # if tracer2 is None, then the assumption is that
    # it is the same as tracer_1.

    # for matter cross correlation, we will use (for now) a
    # PTNumberCounts tracer with b1=1 and all other bias = 0

    # at some point, do we want to store an spline array for these
    # pt power?

    # nonlinear bias cross-terms with IA are currently missing from
    # here and from FASTPT come back to this.

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = Pk2D(a_arr=a_arr,
                 lk_arr=np.log(w.ks),
                 pk_arr=p_pt.T,
                 is_logp=False)

    return pt_pk
