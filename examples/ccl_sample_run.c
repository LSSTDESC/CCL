#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <ccl.h>

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 513
#define PS 0.1
#define NEFF 3.046



int main(int argc,char **argv)
{
  //status flag
  int status =0;

  // Set neutrino masses
  double* MNU;
  double mnuval = 0.;
  MNU = &mnuval;
  ccl_mnu_convention MNUTYPE = ccl_mnu_sum;

  //whether comoving or physical
  int isco=0;

  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NEFF, MNU, MNUTYPE, W0, WA, HH, NORMPS, NS,-1,-1,-1,-1,NULL,NULL, &status);
  //printf("in sample run w0=%1.12f, wa=%1.12f\n", W0, WA);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

  // Compute radial distances (see include/ccl_background.h for more routines)
  printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
  printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_luminosity_distance(cosmo,1./(1+ZD), &status));
  printf("Distance modulus to z = %.3lf is mu = %.3lf Mpc\n",
	 ZD,ccl_distance_modulus(cosmo,1./(1+ZD), &status));


  //Consistency check
  printf("Scale factor is a=%.3lf \n",1./(1+ZD));
  printf("Consistency: Scale factor at chi=%.3lf Mpc is a=%.3lf\n",
	 ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status),
	 ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status), &status));

  // Compute growth factor and growth rate (see include/ccl_background.h for more routines)
  printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
	 ZD, ccl_growth_factor(cosmo,1./(1+ZD), &status),ccl_growth_rate(cosmo,1./(1+ZD), &status));

  // Compute Omega_m, Omega_L, Omega_r, rho_crit, rho_m at different times
  printf("z\tOmega_m\tOmega_L\tOmega_r\trho_crit\trho_m\tRHO_CRITICAL\n");
  double Om, OL, Or, rhoc, rhom;
  for (int z=10000;z!=0;z/=3){
    Om = ccl_omega_x(cosmo, 1./(z+1), ccl_species_m_label, &status);
    OL = ccl_omega_x(cosmo, 1./(z+1), ccl_species_l_label, &status);
    Or = ccl_omega_x(cosmo, 1./(z+1), ccl_species_g_label, &status);
    rhoc = ccl_rho_x(cosmo, 1./(z+1), ccl_species_crit_label, isco, &status);
    rhom = ccl_rho_x(cosmo, 1./(z+1), ccl_species_m_label, isco, &status);
    printf("%i\t%.3f\t%.3f\t%.3f\t%.3e\t%.3e\t%.3e\n", z, Om, OL, Or, rhoc, rhom, RHO_CRITICAL);
  }
  Om = ccl_omega_x(cosmo, 1., ccl_species_m_label, &status);
  OL = ccl_omega_x(cosmo, 1., ccl_species_l_label, &status);
  Or = ccl_omega_x(cosmo, 1., ccl_species_g_label, &status);
  rhoc = ccl_rho_x(cosmo, 1., ccl_species_crit_label, isco, &status);
  rhom = ccl_rho_x(cosmo, 1., ccl_species_m_label, isco, &status);
  printf("%i\t%.3f\t%.3f\t%.3f\t%.3e\t%.3e\t%.3e\n", 0, Om, OL, Or, rhoc, rhom, RHO_CRITICAL);

  // Compute sigma8
  printf("Initializing power spectrum...\n");
  printf("sigma8 = %.3lf\n\n", ccl_sigma8(cosmo, &status));

  //Create tracers for angular power spectra
  double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    bz_arr[i]=1+z_arr_gc[i];
    z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
    nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
  }
  double *a_arr_resample=(double *)malloc(2*NZ*sizeof(double));
  double *nz_resampled=(double *)malloc(2*NZ*sizeof(double));
  for(int i=0;i<2*NZ;i++) {
    double z=(i+0.5)/NZ;
    a_arr_resample[i]=1./(1+z);
  }

  //CMB lensing tracer
  CCL_ClTracer *ct_cl=ccl_cl_tracer_cmblens(cosmo,1100.,&status);

  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_simple(cosmo,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr, &status);

  //Cosmic shear tracer
  CCL_ClTracer *ct_wl=ccl_cl_tracer_lensing_simple(cosmo,NZ,z_arr_sh,nz_arr_sh, &status);
  printf("ell C_ell(c,c) C_ell(c,g) C_ell(c,s) C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");

  //This function allows you to retrieve some of the tracer's internal functions of redshift
  ccl_get_tracer_fas(cosmo,ct_gc,2*NZ,a_arr_resample,nz_resampled,CCL_CLT_NZ,&status);

  int ells[NL];
  double cells_cc_limber[NL];
  double cells_cg_limber[NL];
  double cells_cl_limber[NL];
  double cells_ll_limber[NL];
  double cells_gl_limber[NL];
  double cells_gg_limber[NL];
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;

  double linstep = 40;
  double logstep = 1.15;
  double dchi = (ct_gc->chimax-ct_gc->chimin)/1000.; // must be below 3 to converge toward limber computation at high ell
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *w=ccl_cl_workspace_default(NL+1,-1,CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);
  ccl_angular_cls(cosmo,w,ct_cl,ct_cl,NL,ells,cells_cc_limber,&status);
  ccl_angular_cls(cosmo,w,ct_cl,ct_gc,NL,ells,cells_cg_limber,&status);
  ccl_angular_cls(cosmo,w,ct_cl,ct_wl,NL,ells,cells_cl_limber,&status);
  ccl_angular_cls(cosmo,w,ct_gc,ct_gc,NL,ells,cells_gg_limber,&status);
  ccl_angular_cls(cosmo,w,ct_gc,ct_wl,NL,ells,cells_gl_limber,&status);
  ccl_angular_cls(cosmo,w,ct_wl,ct_wl,NL,ells,cells_ll_limber,&status);


  for(int l=2;l<=NL;l*=2)
    printf("%d %.3lE %.3lE %.3lE %.3lE %.3lE %.3lE | %.3lE\n",l,cells_cc_limber[l],cells_cg_limber[l],cells_cl_limber[l],cells_gg_limber[l],cells_gl_limber[l],cells_ll_limber[l],cells_gl_limber[l]/sqrt(cells_gg_limber[l]*cells_ll_limber[l]));
  printf("\n");

  //Free up tracers
  ccl_cl_tracer_free(ct_cl);
  ccl_cl_tracer_free(ct_gc);
  ccl_cl_tracer_free(ct_wl);

  // Free arrays
  free(a_arr_resample);
  free(nz_resampled);

  //Halo mass function
  printf("M\tdN/dlog10M(z = 0, 0.5, 1))\n");
  for(int logM=9;logM<=15;logM+=1) {
    printf("%.1e\t",pow(10,logM));
    for(double z=0; z<=1; z+=0.5) {
      printf("%e\t", ccl_massfunc(cosmo, pow(10,logM),1.0/(1.0+z), 200., &status));
    }
    printf("\n");
  }
  printf("\n");

  //Halo bias
  printf("Halo bias: z, M, b1(M,z)\n");
  for(int logM=9;logM<=15;logM+=1) {
    for(double z=0; z<=1; z+=0.5) {
      printf("%.1e %.1e %.2e\n",1.0/(1.0+z),pow(10,logM),ccl_halo_bias(cosmo,pow(10,logM),1.0/(1.0+z), 200., &status));
    }
  }

  //Always clean up!!
  ccl_cosmology_free(cosmo);

  return 0;
}
