_docstring_extra_parameters = \
    """
    Currently supported models for `extra_parameters` are listed below,
    along with a list of keys they accept, and their default values,
    if they have a default.

    * ``'camb'`` ::

        'camb': {'halofit_version': 'mead2020',
                 'HMCode_A_baryon': None,
                 'HMCode_eta_baryon': None,
                 'HMCode_logT_AGN': None,
                 'kmax': 10.0,
                 'lmax': 5000,
                 'dark_energy_model': 'fluid'}

    .. note:: Consult the CAMB documentation for their usage.

    * ``'bcm'`` ::

        'bcm': {'log10Mc': log10(1.2e+14),
                'etab': 0.5,
                'ks': 55.0}

    .. note:: BCM stands for the 'baryonic correction model' of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    * ``'halo_model'`` ::

        'halo_model': {'mass_def': '200m',
                       'mass_def_strict': False,
                       'mass_function': 'Tinker10',
                       'halo_bias': 'Tinker10',
                       'concentration': 'Duffy08'}

    .. note:: Using the halo model to compute the non-linear power spectrum
              is not recommended, due to the assumptions that go into it.
              For generic halo model functionality refer to the `halos`
              sub-package. If 'halo_bias' is not specified and 'mass_function'
              is one of ['Tinker10', 'Sheth99'], the 'halo_bias' will be
              coupled to the 'mass_function'.

    * ``'bacco'`` ::

        'bacco': {'M_c': 14,
                  'eta': -0.3,
                  'beta': -0.22,
                  'M1_z0_cen': 10.5,
                  'theta_out': 0.25,
                  'theta_inn': -0.86,
                  'M_inn': 13.4}

    .. note:: 'BACCO' is the the suite of power spectrum emulators described
              in Arico et al. (2021b), :arXiv:`2104.14568`. Extra parameters
              are passed into the baryon model described in Arico et al.
              (2021a), :arXiv:`2011.15018`.

    * ``'PlanckMG'`` ::

        'PlanckMG': {'c1': 1.0,
                     'c2': 1.0,
                     'lambda': 0.0}

    .. note:: 'PlanckMG' refers to the choice of functional modified gravity
              parametrization in Eqs. 46 & 47 if Ade et al. (2015),
              :arXiv:`1502.01590`, where functions f1 and f2 are set equal to
              the commonly used ratio of the dark energy density parameter at
              some scale factor over the dark energy density parameter today.

    """
