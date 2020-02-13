import numpy as np
from scipy.interpolate import interp1d
from .pyutils import _check_array_params


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
