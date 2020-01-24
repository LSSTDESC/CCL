import numpy as np
from scipy.interpolate import interp1d
from .pyutils import _check_array_params
import fastpt as fpt
from .pk2d import Pk2D


def PTTracer(object):
    def __init_(self):
        self.biases = {}
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


def PTNumberCountsTracer(PTTracer):
    def __init__(self, b1, b2=None, bs=None):
        self.biases = {}

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2
        self.biases['b2'] = self._get_bias_function(b2)
        # Initialize bs
        self.biases['bs'] = self._get_bias_function(bs)


def PTIntrinsicAlignmentTracer(PTTracer):
    def __init__(self, c1, c2=None, cdelta=None):

        self.biases = {}

        # Initialize b1
        self.biases['c1'] = self._get_bias_function(c1)
        # Initialize b2
        self.biases['c2'] = self._get_bias_function(c2)
        # Initialize bs
        self.biases['cdelta'] = self._get_bias_function(cdelta)


def get_pt_pk(cosmo, k, a, tracer_1, tracer_2=None,
              pt_object=None, return_pt_object=False):
    # First initialize the pt_object if needed
    if pt_object is None:
        pt_object = fpt.FASTPT(ks,to_do=to_do,
                               low_extrap=low_extrap,
                               high_extrap=high_extrap,
                               n_pad=n_pad)

    # Now compute the Pk however it is that FASTPT does it

    # I've initialized the tracers to None, assuming that,
    # if tracer_2 is None, then the assumption is that
    # it is the same as tracer_1.

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.

    pt_pk = Pk2D(a_arr=a_array,
                 lk_arr=logk_array,
                 pk_arr=pk_array,
                 is_logp=True)  # ?
    # (see pk2d.py for other options).

    return pt_pk
