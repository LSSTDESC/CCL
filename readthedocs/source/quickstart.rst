.. _quickstart:

**********
Quickstart
**********

CCL is structured around :class:`~pyccl.core.Cosmology` objects which hold the cosmological
parameters and any tabulated data associated with a given cosmology. The
library then provides functions to compute specific quantities of interest.
See the full API documentation through the :mod:`pyccl` module and submodules for more details.

Further, CCL follows the following conventions:

  - all units are non-h-inverse (e.g., Mpc as opposed to Mpc/h)
  - the scale factor ``a`` is preferred over redshift ``z`` as a time coordinate.
  - the :class:`~pyccl.core.Cosmology` object always comes first in
    most function calls
  - argument ordering for power spectra is ``(k, a)``

This example computes the comoving distance and HALOFIT non-linear
power spectrum using the BBKS transfer function:

.. code-block:: python

   >>> import pyccl
   >>> cosmo = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                               h=0.7, n_s=0.95, sigma8=0.8,
                               transfer_function='bbks')
   >>> pyccl.sigma8(cosmo)  # get sigma8
   0.7999999999999998
   >>> z = 1
   >>> pyccl.comoving_radial_distance(cosmo, 1./(1+z))  # comoving distance to z=1 in Mpc
   3303.5261651302458
   >>> pyccl.nonlin_matter_power(cosmo, k=1, a=0.5)  # HALOFIT P(k) at k,z = 1,1
   143.6828250598087

See :ref:`models` for more details on the supported models for various cosmological
quantities (e.g., the power spectrum) and the specification of the cosmological parameters.

A comprehensive set of examples showcasing the different types of functionality
implemented in CCL can be found `here <https://github.com/LSSTDESC/CCLX>`_.
