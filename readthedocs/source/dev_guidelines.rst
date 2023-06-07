.. _code_guidelines:

*********************************************
Code Conventions and Documentation Guidelines
*********************************************

This section provides essential recommendations and best practices for writing
clean, maintainable, and efficient code when contributing to CCL. Following
these guidelines not only ensures consistency across the codebase but also
enhances collaboration and makes it easier for developers to understand and
contribute to the project.

Here you will find a set of dos and don'ts accompanied by illustrative code
examples. So, let's dive in and explore the recommended coding style and best
practices that will make your contributions shine.

Happy coding!


Effective Coding Practices
==========================


Positional & Keyword Arguments
------------------------------
Code is read more than it is written. When extending the API, make sure to
specify keyword-only (KWO) arguments as needed, in order to make the code more
user-friendly.

.. code-block:: python

    # DON'T
    class Cosmology:

        def __init__(Omega_c, Omega_b, h, n_s, sigma8, ...):
            ...

    # hard to tell what each number is
    ccl.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81)


    # DO
    class Cosmology:

        def __init__(*, Omega_c, Omega_b, h, n_s, sigma8, ...):
            ...

    # all arguments keyword-only
    ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81)

In general, only keep as positional-or-keyword (POK) arguments those that are
essential for the function, and add a ``*`` right after, to separate POK and
KWO arguments. Another example is :func:`~pyccl.background.rho_x`.


Extra Parameters in Functions
-----------------------------
It is common for a function to accept a basic set of parameters directly
related to that function, and an extra set of 'configuration' or 'settings'
parameters. Avoid collecting all extra parameters in a dictionary in the
function signature. Instead, use dictionary unpacking for any additional
parameters.

.. code-block:: python

    # DON'T
    class HaloProfileHOD:

        def __init__(self, *, M0, M1, Mmin, fc, fftlog_params: dict):
            ...

    # has to be called like this
    fftlog = {"padding_lo": 0.1, "padding_hi": 10, n_per_decade=100}
    prof = HaloProfileHOD(M0=10, M1=13, Mmin=10, fc=0.8, fftlog_params=fftlog)

    # DO
    class HaloProfileHOD:

        def __init__(self, *, M0, M1, Mmin, fc, **fftlog_params):
            ...

    # parameters can be passed directly
    prof = HaloProfileHOD(M0=10, M1=13, Mmin=10, fc=0.8,
                          padding_lo=0.1, padding_hi=10, n_per_decade=100)


Consistent Naming
-----------------
When extending the API make sure that the names you give to functions and
arguments are consistent with the rest of CCL. Remember, functions generally
operate on some argument, so their name should start with a verb.

.. code-block:: python

    # DON'T
    def normalization(self, cosmology, sf, mdef):
        """Compute the normalization."""
        # `normalization` makes it sound like a property of `self`.
        ...

    # DO
    def get_normalization(self, cosmo, a, *, mass_def):
        """Compute the normalization."""
        # Denote cosmology: `cosmo`, scale factor: `a`, mass definition: `mass_def`.
        # The verb makes it clear there is something to compute.
        ...

    # DO
    def compute_normalization(self, cosmo, a, *, mass_def):
        # Another good name.
        ...


Boilerplate Code
----------------
Think of whether your new code fits better in some already-existing code.
Instead of polluting the namespace with many functions containing boilerplate
code, the best practice is to consolidate them into a single function.

.. code-block:: python

    # DON'T
    def correlation_3dRsd(cosmo, *, r, a, mu, beta, p_of_k_a, use_spline=True):
        """Compute 3D RSD correlation in r/mu coordinates."""
        # boilerplate code
        ...

    def correlation_3dRsd_avgmu(cosmo, *, r, a, beta, p_of_k_a):
        """Compute 3D RSD correlation and average over mu."""
        # boilerplate code
        ...

    def correlation_3dRsd_pi_sigma(cosmo, *, pi, sigma, a, beta, p_of_k_a):
        """Compute 3D RSD correlation in pi/sigma coordinates."""
        # boilerplate code
        ...

    # DO
    def correlation3D_RSD(cosmo, *, r=None, mu=None, pi=None, sigma=None,
                          a, beta, p_of_k_a, use_spline=True):
        # no boilerplate; write only once
        ...

        r, mu = convert_to_r_mu(pi, sigma)

        if mu is None:
            # compute the average over mu
            return ...

        # compute the usual 3D RSD correlation
        return ...

In general, if you find yourself repeating sizeable chunks of code, it is a
hint that you would be better off combining your code into a single body.


The Import System
-----------------
To simplify imports, CCL imports everything from the submodules of each
(sub)package (``from my_module import *``) in the ``__init__.py`` files.
This means that new API has to be declared in ``__all__``. Note that the order
the exposed API appears in the documentation is determined by ``__all__``.

.. code-block:: python
    :caption: my_module.py

    __all__ = ("func1", "func2", "func3",)

    def func3: ...
    def func1: ...
    def func2: ...

.. code-block:: python
    :caption: __init__.py

    from my_module import *

will show the functions in order (1/2/3) in the documentation.

Module-level imports are divided into 3 blocks, with each block in alphabetic
order.

.. code-block:: python

    from __future__ import annotations  # __future__ imports first

    __all__ = (...,)  # expose API as needed

    # 1st block: Standard library.
    import functools
    import os
    import sys
    from inspect import signature
    from typing import Callable, Union

    # 2nd block: Third-party libraries.
    import numpy as np
    from scipy.integrate import simpson

    # 3rd block: Local imports (from outer to inner directories).
    from .. import Cosmology
    from . import MassFunc

If the function you would like to use is also a method of
:class:`~pyccl.cosmology.Cosmology`, avoid importing the function.

.. code-block:: python

    # DON'T
    from ..background import rho_x

    rho_x(cosmo, a)

    # DO
    cosmo.rho_x(a)

Cyclic imports can most of the time be overcome by amending the import order of
the submodules in ``__init__.py``.


Object Overuse
--------------
Python is an object-oriented language, and it's often easy to fall into the
trap of needlessly creating objects.

.. code-block:: python

    class MassDef:

        def __init__(self, Delta, rho_type):
            ...

    # DON'T
    class MassDef200c(MassDef):

        def __init__(self):
            self.Delta = 200
            self.rho_type = "critical"

    # DO
    def MassDef200c():
        # mass definition factory
        return MassDef(200, "critical")

    # EVEN BETTER
    MassDef200c = MassDef(200, "critical")  # since there can only be one 200c

In general, if your subclass only overrides :meth:`__init__` this is a hint
that you probably don't need a subclass. Another example is
:func:`~pyccl.cosmology.CosmologyVanillaLCDM`, which is a function, rather than
a subclass of :class:`~pyccl.cosmology.Cosmology`.


Flag Overuse
------------
In functional languages (such as C) it is common to use (boolean) flags for
type control. This is hardly ever the case with object-oriented languages, and
must be avoided.

.. code-block:: python

    # DON'T
    class HaloProfile:
        _is_Pressure = False

    class HaloProfilePressureGNFW(HaloProfile):
        _is_Pressure = True

    # type-checking
    HaloProfilePressureGNFW()._is_Pressure

    # DO
    class HaloProfile: ...

    class HaloProfilePressure(HaloProfile): ...

    class HaloProfilePressureGNFW(HaloProfilePressure): ...

    # type-checking
    isinstance(HaloProfilePressureGNFW(), HaloProfilePressure)

Flags can almost always be refactored with inheritance and/or
composition. If you find that you have to use them, this is a hint that your
inheritance structure is flawed. Type checks ought to use :func:`isinstance`
instead of accessing internal object attributes/properties.

In general, only use flags for dynamic attributes. Attributes which represent
fixed constants for objects should not be flags.


Early Returns
-------------
Instead of using multiple nested ``if-elif-else`` blocks for flow control, try
to deal with special cases first, and return early. Then, the rest of the code
can be unindented. This makes it much easier to read and maintain.

.. code-block:: python

    # DON'T
    def my_func():
        ...

        if a is None:
            return a_arr, s2B_arr
        else:
            if np.ndim(a) == 0:
                return s2B_arr[0]
            else:
                return s2B_arr

    # DO
    def my_func():
        ...
        if a is None:
            return a_arr, s2B_arr
        if np.ndim(a) == 0:
            return s2B_arr[0]
        return s2B_arr

A similar strategy (raising early) can be followed with exceptions. This way,
errors are isolated from code and the codebase is kept tidy.

Also, avoid defining local variables which are immediately returned; instead,
return the expression directly.


Documenting Code
================

Documentation is essential for code development. Here are two key reasons why
it matters:

#. **Reference for Users**: Documentation serves as a reliable reference for
   users, providing clear explanations of the API, functions, and classes. It
   helps users understand how to effectively use the code, saving them time and
   effort.

#. **Troubleshooting and Support**: Documentation is the first point of contact
   when issues arise. It offers troubleshooting tips, known issues, and
   solutions, empowering users to resolve problems independently.

Prioritizing documentation enhances the user experience, promotes
self-sufficiency, and improves the reliability of CCL.

This paragraph describes the best practices for documenting the code. CCL
uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and largely follows the
`Numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
for documenting code. A comprehensive explanation of what Sphinx can do
(with emphasis on the Directives and the Roles) can be found in the
`reStructured Text (reST) primer
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
of the Sphinx docs.


Docstrings
----------
CCL is a scientific package and it is common for docstrings to have equations.
Prepend the docstring with an ``r`` to avoid multiple backslashes in math-mode.

.. code-block::

    # These two are equivalent.
    """Did you know that :math:`\\sin^2(x) + \\cos^2(x) = 1`?"""
    r"""Did you know that :math:`\sin^2(x) + \cos^2(x) = 1`?"""

The language used in docstrings (as well as warning and error messages) should
be simple and consise. Avoid filler words and information that is not directly
related to the message you are trying to convey.

.. code-block::

    # DON'T
    raise ValueError("The range of the internal splines of the Pk2D object "
                     "that you passed do not match the range of the internal "
                     "splines stored in cosmo. Please use the the same range.")

    # DO
    raise ValueError("Range mismatch with splines of `cosmo`.")

.. code-block::

    # DON'T
    def comoving_radial_distance(cosmo, a):
        """Calculates the comoving radial distance given a specific cosmology
        and some scale factor.
        """

    # DO
    def comoving_radial_distance(cosmo, a):
        """Compute the comoving radial distance."""

Other points to remember:

* Avoid using 3rd-person to describe what a function/class does.

* Do not leave a blank space in the beginning of the docstring.

* Start docstring with a single-line consise description of what the function
  does. Write the rest in a separate paragraph.

* Remember to add a "Returns" and "Raises" section if functions return/raise.
  Example in :func:`~pyccl.background.angular_diameter_distance`.

* Always provide units (preferrably in roman font in math-mode) when describing
  physical quantities.

* You may make private or dunder functions/methods show in the documentation
  by adding the ``:meta public:`` info field. We only recommend that you do this in
  abstract classes or other special cases where exposing a private function is
  appropriate. All :meth:`__call__` methods are automatically exposed. See the
  `autodoc documentation
  <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ for
  details.


Type Hints
----------
CCL uses the extensions ``sphinx.ext.autodoc`` and ``sphinx_autodoc_typehints``.
Type-hinting is optional, but strongly encouraged, as it accomplishes 3 things:

#. Declutters the docstring, because the build automatically picks up the
   argument types from the code and adds them to the docstring;

#. Makes manual type-checks in the code redundant;

#. Makes the code more robust, as static typing can be enforced with tools like
   `mypy <https://mypy-lang.org/>`_.


Examples of type hints at varied levels of complexity can be found in
:func:`~pyccl.background.angular_diameter_distance`, :class:`~pyccl.pk2d.Pk2D`,
and :class:`~pyccl.halos.halo_model.HMCalculator`. Refer to the `typing
documentation <https://docs.python.org/3/library/typing.html>`_ for details.
The most common type hints are ``Union``, ``Optional``, ``Literal``, ``Tuple``,
``List``, ``Sequence``, ``Iterable``, ``Callable``, ``Any``
(from :mod:`typing`) and ``NDArray`` (from :mod:`numpy.typing`).

To type hint with local types you need to add this line at the top of the
module:

.. code-block:: python

    from __future__ import annotations

in order to enable forward references with `postponed evaluation of annotations
<https://peps.python.org/pep-0563/>`_.

To avoid cyclic imports when importing classes only used for type-hinting
you can use `TYPE_CHECKING
<https://peps.python.org/pep-0484/#runtime-or-type-checking>`_ like so:

.. code-block:: python
    :caption: halos/massdef.py

    from typing import TYPE_CHECKING

    # `cosmology.py` is imported after `massdef.py`, so attempting to
    # import it here will result to a cyclic import. However, we want to
    # make it available to the type checkers.
    if TYPE_CHECKING:
        from .. import Cosmology

An exception is `array_like`, which is the type of anything that can be cast to
a Numpy array: numbers (integers, floats), sequences (lists, tuples),
arrays etc. This kind of I/O is ubiquitous in CCL, and we recommend that the
dimensionality of the `array_like` object be specified in the style of `Scipy
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fht.html>`_.

In the following example from :class:`~pyccl.pk2d.Pk2D` we have specified the
input and output dimensions of the `array_like` objects, in a way that is
self-explanatory, so we don't have to add extra information in the description.

.. code-block:: python
    :caption: pk2d.py

    from typing import Union

    from numpy.typing import ArrayLike


    class Pk2D:

        def __call__(
                self,
                k: ArrayLike,
                a: ArrayLike,
                cosmo: Optional[Cosmology] = None,
                *,
                derivative: bool = False
        ) -> ArrayLike:
            r"""Evaluate the power spectrum or its logarithmic derivative.

            Arguments
            ---------
            k : array_like (nk,)
                Wavenumber (in :math:`\rm Mpc^{-1}`).
            a : array_like (na,)
                Scale factor.
            cosmo
                Cosmological parameters. Used to evaluate the power spectrum
                outside of the interpolation range in `a` (thorugh the linear
                growth). If None, out-of-bounds queries raise an exception.
            derivative
                Whether to evaluare the logarithmic derivative,
                :math:`\frac{{\rm d}\log P(k)}{{\rm d}\log k}`.

            Returns
            -------
            array_like (na, nk)
                Evaluated power spectrum.
            """

.. note::

    At the moment of writing these guidelines (March 2023) ``ArrayLike`` is not
    automatically recognized by ``sphinx_autodoc_typehints`` and we have
    used ``Union[Real, NDArray[Real]]`` for input `array_like` and
    ``Union[float, NDArray[float]]`` for output `array_like` instead.


Attributes
----------
It is generally helpful to list the important attributes of an object and
provide a brief description. You may either include an "Attributes" section in
the docstring, or (better yet) include them in the code, as undeclared class
variables:

.. code-block:: python
    :caption: halos/profiles/profile_base.py

    class HaloProfile:
        r"""Abstract base class for halo profiles.

        Parameters
        ----------
        mass_def
            Mass definition.
        concentration
            Mass-concentration relation, used to calculate the scale radius in
            some halo profiles.
        """
        mass_def: MassDef
        concentration: Union[Concentration, None]
        precision_fftlog: FFTLogParams
        "FFTLog accuracy parameters."

        def __init__(self, *, mass_def, concentration, ...):
            self.mass_def = ...
            self.concentration = ...
            self.precision_fftlog = ...

Note that we have only included a docstring to `precision_fftlog` as the other
two already have a description in the "Parameters" section.

Properties are 'special' attributes, and they should appear right after
:meth:`__init__` (remember that documentation follows the order of the source).


Module/Subpackage Summaries
----------------------------
New modules should contain a brief description of the contents, at the top:

.. code-block::
    :caption: background.py

    """
    ====================================
    Background (:mod:`pyccl.background`)
    ====================================

    Functions to compute background quantities: distances, energies, growth.
    """

Likewise, include a brief description in ``__init__.py`` of new subpackages:

.. code-block::
    :caption: __init__.py

    """
    ==============================
    Baryons (:mod:`pyccl.baryons`)
    ==============================

    Add baryonic effects to power spectra.
    """

Cross-Referencing
-----------------
You can refer to other objects (internal or external) using the `Sphinx roles
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html>`_.
These show as clickable items in the build. You may either specify the full
path (e.g. ``:class:`~pyccl.halos.massdef.MassDef```) or, if already imported,
refer to the object directly (e.g. ``:class:`~MassDef```).


References
----------
CCL uses ``sphinxcontrib.bibtex`` as the docstring referencing tool. The list
of references is in ``readthedocs/refs.bib``. When a new reference is added
there, it can be used in the docstrings:

.. code-block:: python
    :caption: halos/profiles/einasto.py

    class HaloProfileEinasto(HaloProfileMatter):
        r"""Halo profile by :footcite:t:`Einasto65`.

        References
        ----------
        .. footbibliography::
        """

``:footcite:t:`` adds the full text (in this case ``'Einasto [1]'``), while
``:footcite:p:`` only adds a superscript reference to the bibliography.
Remember to add ``.. footbibliography::`` in its own "References" section
(usually at the end of the docstring).
