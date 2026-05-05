pyccl.baryons.spk module
=========================

SP(k) references
----------------

- Salcido et al. 2023, MNRAS 523, 2247:
   https://doi.org/10.1093/mnras/stad1474
- arXiv preprint: https://arxiv.org/abs/2305.09710
- pyspk implementation: https://github.com/jemme07/pyspk

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
- Runnable example scripts are available in:
   https://github.com/jemme07/pyspk/tree/main/examples

SP(k) mode equations (summary)
------------------------------

The multiplicative suppression always follows
``P_bar(k, a) = P_DMO(k, a) * f_SPk(k, a)``. The relation mode controls
the baryon-fraction parameterization used by ``pyspk``:

- ``power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = a\left(\frac{M_{\mathrm{SO}}}{M_{\mathrm{pivot}}}\right)^b

  with :math:`a=\mathrm{fb\_a}`, :math:`b=\mathrm{fb\_pow}`,
  :math:`M_{\mathrm{pivot}}=\mathrm{fb\_pivot}` (optional).

- ``cosmo_power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = \frac{\exp(\alpha)}{100}
     \left(\frac{M_{500c}}{10^{14}\,M_\odot}\right)^{\beta-1}
     \left(\frac{E(z)}{E(0.3)}\right)^\gamma

- ``double_power_law``:

  .. math::

     \frac{f_b}{\Omega_b/\Omega_m} = \frac{1}{2}\,\epsilon
     \left[\left(\frac{M_{500c}}{M_{\mathrm{pivot}}}\right)^\alpha +
     \left(\frac{M_{500c}}{M_{\mathrm{pivot}}}\right)^\beta\right]
     \left(\frac{E(z)}{E(0.3)}\right)^\gamma

  with :math:`M_{\mathrm{pivot}}=\mathrm{m\_pivot}`.

- ``binned``:
   Provide tabulated halo-mass and baryon-fraction samples through
   ``M_halo`` and ``fb``. In :class:`pyccl.BaryonsSPK`, these are passed as
   arrays in ``relation_params`` (same length, representing
   :math:`f_b(M, z)` at the evaluation redshift). Set ``extrapolate=True``
   to allow extrapolation beyond the tabulated mass range; otherwise values
   outside the tabulated range are treated according to ``pyspk`` behavior.

For exact mode definitions, calibration domain, and edge-case handling,
refer to the upstream ``pyspk`` documentation (authoritative source):
https://github.com/jemme07/pyspk

.. automodule:: pyccl.baryons.spk
   :members:
   :undoc-members:
   :show-inheritance:
