import numpy as np
from .base import _to_hashable, hash_
from .pyutils import _get_spline1d_arrays


class Table:
    """Build nice tables.

    Comments describing the capabilities of each method are included below.
    """

    def __init__(self, *, n_y=6, n_x=6, decimals=2,
                 legend="", newline="\n\t", missing="...",
                 data_x=None, data_y=None, data_z=None, meta=[]):
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.meta = meta
        self.n_y = n_y
        self.n_x = n_x
        self.newline = newline
        self.missing = missing
        self.legend = legend
        self.entryl = 7 + decimals
        self.div = max(self.entryl, len(self.legend))
        self.form = f"{self.entryl}.{decimals}e"
        self.idx = np.arange(self.n_y//2).tolist() + \
            np.arange(-self.n_y//2, 0).tolist()

    def divider(self, new_line: bool):
        # Horizontal line in Table.
        # +=============+=======================+
        divider = f"+{'=' * (self.div+2)}+"
        divider += f"{'='*(1+(self.n_x*(self.entryl+1))+len(self.missing)+1)}+"
        divider += new_line * f"{self.newline}"
        return divider

    def wrap(self, expr, a, b=None):
        # Wrap the expression with `a` at the beginning and `b` at the end.
        return f"{a}{expr}{b if b is not None else a}"

    def print_left(self, expr):
        # Print the left part of a row of the Table (see `fullrow`).
        return self.wrap(self.wrap(f"{expr:{self.div}}", " "), "|")

    def print_elements(self, arr):
        # Print the elements of `arr`.
        return " ".join([f"{i:{self.form}}" for i in arr])

    def print_right(self, arr, num):
        # Print the right part of a row of the Table (see `fullrow`).
        s = self.wrap(self.print_elements(arr[:num//2]), " ")
        s += f"{self.missing}"
        s += self.wrap(self.wrap(
            self.print_elements(arr[-num//2:]),
            " "), "", f"|{self.newline}")
        return s

    def fullrow(self, s1, s2, num):
        # Print a full row of the Table.
        # |   1.00e-02  | 1.71e-01 ... 3.31e-05 |
        s = self.print_left(f"{s1:{'' if isinstance(s1, str) else self.form}}")
        s += self.print_right(s2, num)
        return s

    def missing_row(self):
        # Print row with skipped values.
        # |  ...   |            ...             |
        s = self.wrap(self.wrap(f"{self.missing:^{self.div}}", " "), "|")
        length = len(self.divider(new_line=False)) - len(s) - 3
        s += self.wrap(self.wrap(
            f"{self.missing:^{length}}",
            " "), "", f"|{self.newline}")
        return s

    def metadata(self):
        # If an object carries metadata that need to be included in `__repr__`
        # pass them here in a list. Each element will start a new row.
        s = ""
        for m in self.meta:
            s += self.wrap(self.wrap(
                f"{m:<{len(self.divider(new_line=False)) - 4}}",
                " "), "|") + self.newline
        return s

    def build(self):
        # Build the table.
        s = self.divider(new_line=True)
        s += self.fullrow(f"{self.legend}", self.data_x, self.n_x)
        s += self.divider(new_line=True)
        s += "".join([self.fullrow(self.data_y[i], self.data_z[i], self.n_x)
                      for i in self.idx[:self.n_y//2]])
        s += self.missing_row()
        s += "".join([self.fullrow(self.data_y[i], self.data_z[i], self.n_x)
                      for i in self.idx[-self.n_y//2:]])
        s += self.divider(new_line=bool(self.meta))
        s += self.metadata()
        s += self.divider(new_line=False) if self.meta else ""
        return s


def _build_string_Cosmology(self):
    """Build the ``Cosmology`` representation.

    Cosmology equivalence is tested via its representation. Therefore,
    there is limiting behavior where ``'=='`` will return ``False``
    even though the compared cosmologies return the same theoretical
    predictions. This happens whenever:
        - Exactly one Cosmology is an instance of ``CosmologyCalculator``.
        - Cosmologies defined with different parameter sets, where one can
          be computed from the other (e.g. ``sigma8`` and ``A_s``).
        - Instances of ``CosmologyCalculator`` which do not contain exactly
          the same linear & non-linear power spectrum entries.

    Example output ::

        <pyccl.core.Cosmology>
            Omega_b = 0.05
            Omega_c = 0.25
            h       = 0.67
            n_s     = 0.96
            sigma8  = 0.81
            extra_parameters =
                test = {'param': 18.4}
            HASH_ACCURACY_PARAMS = 0x1959cbc9
            HASH_PK = 0xbca03ab0
    """
    newline = "\n\t"
    cls = self.__class__

    def test_eq(key, val, default):
        # Neutrino masses can be a list, so use `np.all` for comparison.
        # `np.all` is expensive, so only use that with `m_nu`.
        if key not in ["m_nu", "z_mg", "df_mg"]:
            return val == default
        return np.all(val == default)

    def printdict(dic):
        # Print the non-default parameters listed in a parameter dictionary.
        base = cls.__base__ if cls.__qualname__ != "Cosmology" else cls
        params = base._init_signature.parameters
        defaults = {param: value.default for param, value in params.items()}
        dic = {key: val for key, val in dic.items()
               if not test_eq(key, val, defaults.get(key))}
        dic.pop("extra_parameters", None)
        if not dic:
            return ""
        length = max(len(key) for key, val in dic.items())
        tup = _to_hashable(dic)
        s = ""
        for param, value in tup:
            s += f"{newline}{param:{length}} = {value}"
        return s

    def printextras(dic):
        # Print any extra parameters.
        if dic["extra_parameters"] is None:
            return ""
        tup = _to_hashable(dic["extra_parameters"])

        s = f"{newline}extra_parameters ="
        for key, value in tup:
            s += f"{newline}\t{key} = {dict(value)}"
        return s

    def metadata():
        # Print hashes for the accuracy parameters and the stored Pk2D's.
        H = hex(hash_(self._accuracy_params))
        s = f"{newline}HASH_ACCURACY_PARAMS = {H}"
        if self.__class__.__qualname__ == "CosmologyCalculator":
            # only need the pk's if we compare CosmologyCalculator objects
            H = 0
            if self._has_pk_lin:
                H += sum([hash_(pk) for pk in self._pk_lin.values()])
            if self._has_pk_nl:
                H += sum([hash_(pk) for pk in self._pk_nl.values()])
            H = hex(H)
            s += f"{newline}HASH_PK = {H}"
        return s

    s = "<pyccl.core.Cosmology>"
    s += printdict(self._params_init_kwargs)
    s += printdict(self._config_init_kwargs)
    s += printextras(self._params_init_kwargs)
    s += metadata()
    return s


def _build_string_Pk2D(self, na=6, nk=6, decimals=2):
    """Build the ``Pk2D`` representation.

    Example output ::

        <pyccl.Pk2D>
            +===============+=============================================+
            | a \\ log10(k) | -4.30e+00 -4.16e+00 ...  9.29e-01  1.02e+00 |
            +===============+=============================================+
            |   1.00e-02    |  1.71e-01  2.36e-01 ...  5.82e-05  3.31e-05 |
            |   1.26e-02    |  2.78e-01  3.82e-01 ...  9.15e-05  5.21e-05 |
            |      ...      |                     ...                     |
            |   9.77e-01    |  1.14e+03  1.57e+03 ...  3.31e-01  1.88e-01 |
            |   1.00e+00    |  1.17e+03  1.60e+03 ...  3.39e-01  1.93e-01 |
            +===============+=============================================+
            | is_log = True , extrap_orders = (1, 2)                      |
            | HASH_ARRS = 0x1d3524ad                                      |
            +===============+=============================================+
    """
    if not self.has_psp:
        return "pyccl.Pk2D(empty=True)"

    # get what's needed from the Pk2D object
    a, lk, pk = self.get_spline_arrays()
    lk /= np.log(10)  # easier to read in log10
    islog = str(bool(self.psp.is_log))
    extrap = (self.psp.extrap_order_lok, self.psp.extrap_order_hik)
    H = hex(sum([hash_(obj) for obj in [a, lk, pk]]))

    newline = "\n\t"  # what to do when starting a new line
    legend = "a \\ log10(k)"  # table legend
    meta = [f"is_log = {islog:5.5s}, extrap_orders = {extrap}"]
    meta += [f"HASH_ARRS = {H:34}"]

    T = Table(n_y=na, n_x=nk, decimals=decimals, legend=legend,
              newline=newline, data_x=lk, data_y=a, data_z=pk, meta=meta)

    s = _build_string_simple(self) + f"{newline}"
    s += T.build()
    return s


def _build_string_simple(self):
    """Simple representation.

    Example output ::

        <pyccl.emulator.Emulator>
    """
    return f"<{self.__module__}.{self.__class__.__qualname__}>"


def _build_string_from_init_attrs(self):
    """Build a generic representation when all `__init__` parameters
    are also instance attributes.

    Example output ::

        <pyccl.halos.halo_model.HMCalculator>
            mass_function = MassFuncTinker08,  HASH = 0xd3b29dd3
            halo_bias = HaloBiasTinker10,  HASH = 0x9da644b5
            mass_def = pyccl.halos.MassDef(Delta=500, rho_type=critical)
    """

    # collect the dictionary of input values different to defaults
    params = self._init_signature.parameters
    defaults = {param: value.default
                for param, value in params.items()
                if param != "self"}

    passed = {param: getattr(self, param) for param in defaults}

    s = _build_string_simple(self)
    dic = {param: value
           for param, value in passed.items()
           if value != defaults[param]}
    if not dic:
        return s

    newline = "\n\t"
    for param, value in dic.items():
        # print the non-default attributes one-by one
        s += f"{newline}{param} = "
        if "\n" in repr(value):
            # if too long, print the type and its hash
            name = value.__class__.__qualname__
            H = hex(hash_(value))
            s += f"{name},  HASH = {H}"
        else:
            s += f"{value}"
    return s


def _build_string_HaloProfile(self):
    """Build a representation for a HaloProfile.

    Example output ::

        <pyccl.halos.profiles.HaloProfileHOD>
            c_m_relation = ConcentrationDuffy08,  HASH = 0xc8d6ef04
            lMmin_0 = 11.5
            lM1_0 = 12.0
            ns_independent = True
            HASH_FFTLOG = 0x5546e5dc
    """
    H = hex(hash_(self.precision_fftlog))
    newline = "\n\t"
    s = _build_string_from_init_attrs(self)
    s += f"{newline}HASH_FFTLOG = {H}"
    return s


def _build_string_Tracer(self):
    """Buld a representation for a Tracer.

    Example output ::

        <pyccl.tracers.CMBLensingTracer>
            cosmo = CosmologyVanillaLCDM,  HASH = 0xda3e0d42
            z_source = 1101
            n_samples = 128
            NUM_TRACERS = 1
            HASH_SPLINES = 0xf4bd2054
    """
    tracers = self._trc
    H = hex(sum([hash_(_get_spline1d_arrays(tr.kernel.spline))
                 for tr in tracers]))
    newline = "\n\t"
    s = _build_string_from_init_attrs(self)
    s += f"{newline}NUM_TRACERS = {len(tracers)}"
    if len(tracers) > 0:
        s += f"{newline}HASH_SPLINES = {H}"
    return s


def _build_string_Tk3D(self, na=2, nk=4, decimals=2):
    """Build a representation for a Tk3D object.

    Example output ::

        <pyccl.Tk3D>
            +================+=============================================+
            | a \\ log10(k1) | -4.00e+00 -3.33e+00 ...  1.33e+00  2.00e+00 |
            +================+=============================================+
            |   5.00e-02     |  4.46e+07  9.62e+06 ...  2.07e+02  4.46e+01 |
            |       ...      |                     ...                     |
            |   1.00e+00     |  2.00e+09  4.30e+08 ...  9.26e+03  2.00e+03 |
            +================+=============================================+
            +================+=============================================+
            | a \\ log10(k2) | -4.00e+00 -3.33e+00 ...  1.33e+00  2.00e+00 |
            +================+=============================================+
            |   5.00e-02     |  4.46e+01  1.78e+00 ...  2.82e-10  1.12e-11 |
            |       ...      |                     ...                     |
            |   1.00e+00     |  2.00e+03  7.94e+01 ...  1.26e-08  5.01e-10 |
            +================+=============================================+
            | is_log = True , extrap_orders = (1, 1)                       |
            | HASH_ARRS = 0x780972f4                                       |
            +================+=============================================+
    """
    if not self.has_tsp:
        return "pyccl.Tk3D(empty=True)"

    # get what's needed from the Tk3D object
    a, lk1, lk2, tks = self.get_spline_arrays()
    lk1 /= np.log(10)  # easier to read in log10
    lk2 /= np.log(10)  # easier to read in log10
    islog = str(bool(self.tsp.is_log))
    extrap = (self.tsp.extrap_order_lok, self.tsp.extrap_order_hik)
    H = hex(sum([hash_(obj) for obj in [a, lk1, lk2, *tks]]))

    newline = "\n\t"
    meta = [f"is_log = {islog:5.5s}, extrap_orders = {extrap}"]
    meta += [f"HASH_ARRS = {H:34}"]

    # we will print 2 tables
    if not self.tsp.is_product:
        # get the start and the end of the trispectrum, diagonally in `k`
        tks = [tks[0][:, 0, :], tks[0][:, :, -1]]

    T = Table(n_y=na, n_x=nk, decimals=decimals, newline=newline,
              data_y=a, legend="a \\ log10(k1)", meta=[])

    s = _build_string_simple(self) + f"{newline}"
    T.data_x, T.data_z = lk1, tks[0]
    s += T.build() + f"{newline}"
    T.legend = "a \\ log10(k2)"
    T.data_x, T.data_z = lk2, tks[1]
    T.meta = meta
    s += T.build()
    return s
