v 0.3 API changes:

C library:

Changes in ccl_core.c:

Old call signature:
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k,
				     double N_nu_rel, double N_nu_mass, double mnu,
				     double w0, double wa, double h, double norm_pk,
				     double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks,
				     int nz_mgrowth,double *zarr_mgrowth,
					 double *dfarr_mgrowth, int *status)
					 
New call signature:
ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k,
				     double Neff, double* mnu, ccl_mnu_type_label mnu_type,
				     double w0, double wa, double h, double norm_pk,
				     double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks,
				     int nz_mgrowth,double *zarr_mgrowth,
				     double *dfarr_mgrowth, int *status)
				 
Old call signature:
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k,
				     double N_nu_rel, double N_nu_mass, double mnu,
				     double w0, double wa, double h, double norm_pk,
				     double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks,
				     int nz_mgrowth,double *zarr_mgrowth,
                     double *dfarr_mgrowth, int *status)
                     
New call signature:
ccl_cosmology * ccl_cosmology_create_with_params(double Omega_c, double Omega_b, double Omega_k,
						 double Neff, double* mnu, ccl_mnu_type_label mnu_type,
						 double w0, double wa, double h, double norm_pk, double n_s,
						 double bcm_log10Mc, double bcm_etab, double bcm_ks,
						 int nz_mgrowth, double *zarr_mgrowth, 
						 double *dfarr_mgrowth, ccl_configuration config,
						 int *status)
						 
(Similar changes apply for any ccl_parameters_create_..._nu convenience functions.)

Python wrapper:

Changes in core.py:

Old call signature:
ccl.Parameters(Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
                 Omega_k=0., N_nu_rel=3.046, N_nu_mass=0., m_nu=0.,w0=-1., wa=0.,
                 bcm_log10Mc=math.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., sigma8=None,
                 z_mg=None, df_mg=None)

New call signature:
ccl.Parameter(Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
              Omega_k=0., Neff = 3.046, m_nu=0., mnu_type = None, w0=-1., wa=0., 
              bcm_log10Mc=math.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., sigma8=None,
              z_mg=None, df_mg=None)
              
Old call signature:
cc.Cosmology(params=None, config=None,
             Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
             Omega_k=0., N_nu_rel=3.046, N_nu_mass=0., m_nu=0., w0=-1., wa=0.,
             bcm_log10Mc=math.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., sigma8=None,
             z_mg=None, df_mg=None, 
             transfer_function='boltzmann_class',
             matter_power_spectrum='halofit',
             baryons_power_spectrum='nobaryons',
             mass_function='tinker10')

New call signature:
ccl.Cosmology(params=None, config=None,
              Omega_c=None, Omega_b=None, h=None, A_s=None, n_s=None, 
              Omega_k=0., Neff=3.046, m_nu=0., mnu_type = None, w0=-1., wa=0.,
              bcm_log10Mc=math.log10(1.2e14), bcm_etab=0.5, bcm_ks=55., sigma8=None,
              z_mg=None, df_mg=None, 
              transfer_function='boltzmann_class',
              matter_power_spectrum='halofit',
              baryons_power_spectrum='nobaryons',
              mass_function='tinker10', emulator_neutrinos='strict')

