pyccl.baryons.spk module
=========================

SP(k) is a simulation-based model for the impact of baryon physics on the
non-linear matter power spectrum (`Salcido et al. 2023
<https://doi.org/10.1093/mnras/stad1474>`_). It predicts the suppression of
power as a function of the mean baryon fraction of haloes, calibrated on the
ANTILLES suite of 400 hydrodynamical simulations spanning a wide feedback
landscape, accurate to approximately percent level for
:math:`k \lesssim 12\,h\,\mathrm{Mpc}^{-1}` and :math:`z \leq 3`.

Python implementation: `pyspk <https://github.com/jemme07/pyspk>`_

Minimum supported upstream version for this wrapper: ``pyspk>=2.0.1``.

Units and Conversion
--------------------

CCL follows non-h-inverse conventions. For :class:`pyccl.BaryonsSPK` this
means users pass :math:`k` in :math:`\mathrm{Mpc}^{-1}`.

Internally, the wrapper converts to ``pyspk`` units
(:math:`h/\mathrm{Mpc}`) via

.. math::

   k\,[h\,\mathrm{Mpc}^{-1}] = k\,[\mathrm{Mpc}^{-1}] / h

This conversion is automatic; users should not pre-convert ``k`` when calling
``pyccl`` APIs.

Halo masses (``fb_pivot``, ``m_pivot``, ``M_halo``) are always in units of
:math:`M_\odot` (solar masses), consistent with ``pyspk``. Baryon-fraction
parameters (``fb_a``, ``fb``) are dimensionless ratios
:math:`f_b/(\Omega_b/\Omega_m)`.

Supported relation modes
------------------------

The :class:`pyccl.BaryonsSPK` wrapper supports four SP(k) relation modes,
matching ``pyspk``:

- ``power_law``: ``fb_a``, ``fb_pow``, optional ``fb_pivot``.
- ``cosmo_power_law``: ``alpha``, ``beta``, ``gamma``.
- ``double_power_law``: ``epsilon``, ``alpha``, ``beta``, ``gamma``,
   ``m_pivot``.
- ``binned``: tabulated ``M_halo`` and ``fb`` values, plus optional
   ``extrapolate``.

Mode details and examples
-------------------------

- Full mode definitions and usage examples are maintained in ``pyspk``:
   https://github.com/jemme07/pyspk
- Runnable ``pyspk`` example notebook:
   https://github.com/jemme07/pyspk/blob/main/examples/pySPk_Examples.ipynb
- CCL integration demo:
   https://github.com/LSSTDESC/CCL/blob/master/examples/spk_demo.ipynb

SP(k) mode equations (summary)
------------------------------

The multiplicative suppression always follows
``P_bar(k, a) = P_DMO(k, a) * f_SPk(k, a)``. The relation mode controls
the baryon-fraction parameterization used by ``pyspk``:

- ``power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = a\left(\frac{M_{\mathrm{SO}}}{M_{\mathrm{pivot}}}\right)^b

  with :math:`a=\mathrm{fb\_a}` (dimensionless normalisation),
  :math:`b=\mathrm{fb\_pow}` (dimensionless slope), and
  :math:`M_{\mathrm{pivot}}=\mathrm{fb\_pivot}` in :math:`M_\odot`
  (optional; default :math:`1\,M_\odot`).

- ``cosmo_power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = \frac{\exp(\alpha)}{100}
     \left(\frac{M_{500c}}{10^{14}\,M_\odot}\right)^{\beta-1}
     \left(\frac{E(z)}{E(0.3)}\right)^\gamma

  with :math:`\alpha` (log-space normalisation), :math:`\beta` (power-law
  slope), and :math:`\gamma` (redshift-evolution exponent); the implicit
  pivot mass is :math:`10^{14}\,M_\odot`.

- ``double_power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = \frac{1}{2}\,\epsilon
     \left[\left(\frac{M_{500c}}{M_{\mathrm{pivot}}}\right)^\alpha +
     \left(\frac{M_{500c}}{M_{\mathrm{pivot}}}\right)^\beta\right]
     \left(\frac{E(z)}{E(0.3)}\right)^\gamma

  with :math:`\epsilon` (dimensionless normalisation of
  :math:`f_b/(\Omega_b/\Omega_m)` at the pivot mass),
  :math:`\alpha`, :math:`\beta` (low- and high-mass slopes),
  :math:`\gamma` (redshift-evolution exponent), and
  :math:`M_{\mathrm{pivot}}=\mathrm{m\_pivot}` in :math:`M_\odot`.

- ``binned``:
   Provide tabulated halo-mass and baryon-fraction samples through
   ``M_halo`` (halo masses in :math:`M_\odot`) and ``fb`` (dimensionless
   baryon fractions :math:`f_b/(\Omega_b/\Omega_m)`, same length as
   ``M_halo``). In :class:`pyccl.BaryonsSPK`, these are passed as arrays
   in ``relation_params`` (same length, representing
   :math:`f_b(M, z)` at the evaluation redshift). Set ``extrapolate=True``
   to allow extrapolation beyond the tabulated mass range; otherwise values
   outside the tabulated range are treated according to ``pyspk`` behavior.

Out-of-range behaviour
----------------------

The SP(k) model is calibrated up to :math:`k_{\max} = 12\,h/\mathrm{Mpc}`
and :math:`z_{\max} = 3`. When a requested :math:`k` or :math:`z` exceeds
the calibrated range, :class:`pyccl.BaryonsSPK` applies a configurable
policy controlled by two constructor parameters:

- ``k_out_of_range`` (default ``"raise"``): governs :math:`k` values
  beyond calibration.
- ``z_out_of_range`` (default ``"unity"``): governs :math:`z` values
  (i.e. low scale-factor :math:`a`) beyond calibration.

Each accepts one of three string values:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Behaviour
   * - ``"raise"``
     - Raise a ``ValueError`` if any requested value exceeds calibration.
   * - ``"unity"``
     - Return a suppression factor of 1.0 (i.e. no baryonic effect) for
       out-of-range entries. Computation proceeds for the valid range.
   * - ``"nan"``
     - Fill out-of-range entries with ``NaN``. This is consistent with
       ``pyspk``'s own treatment of out-of-fitting-limit baryon fractions
       and allows downstream code to mask or interpolate as appropriate.

When ``_include_baryonic_effects`` builds a corrected :class:`~pyccl.Pk2D`,
any resulting ``NaN`` rows (from ``z_out_of_range="nan"``) or columns
(from ``k_out_of_range="nan"``) are automatically dropped with a
:class:`~pyccl.CCLWarning`, ensuring the output spline remains finite.

Example
^^^^^^^

.. code-block:: python

   import pyccl as ccl

   bar = ccl.BaryonsSPK(
       SO=200,
       relation_kind="power_law",
       fb_a=0.4,
       fb_pow=0.3,
       fb_pivot=1e13,
       k_out_of_range="unity",   # unity beyond calibration
       z_out_of_range="nan",     # NaN for z > 3
   )

For exact mode definitions, calibration domain, and edge-case handling,
refer to the upstream ``pyspk`` documentation (authoritative source):
https://github.com/jemme07/pyspk

----

API Reference
-------------

.. automodule:: pyccl.baryons.spk
   :members:
   :undoc-members:
   :show-inheritance:
