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

.. automodule:: pyccl.baryons.spk
   :members:
   :undoc-members:
   :show-inheritance:
