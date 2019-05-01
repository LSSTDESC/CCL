#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


/*----- ROUTINE: dc_NakamuraSuto -----
INPUT: cosmology, scale factor
TASK: Computes the peak threshold: delta_c(z) assuming LCDM.
Cosmology dependence of the critical linear density according to the spherical-collapse model.
Fitting function from Nakamura & Suto (1997; arXiv:astro-ph/9710107).
*/
double dc_NakamuraSuto(ccl_cosmology *cosmo, double a, int *status){

  double Om_mz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  double dc0 = (3./20.)*pow(12.*M_PI,2./3.);
  double dc = dc0*(1.+0.012299*log10(Om_mz));

  return dc;

}

/*----- ROUTINE: Dv_BryanNorman -----
INPUT: cosmology, scale factor
TASK: Computes the virial collapse density contrast with respect to the matter density assuming LCDM.
Cosmology dependence of the virial collapse density according to the spherical-collapse model
Fitting function from Bryan & Norman (1998; arXiv:astro-ph/9710107)
*/
double Dv_BryanNorman(ccl_cosmology *cosmo, double a, int *status){

  double Om_mz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  double x = Om_mz-1.;
  double Dv0 = 18.*pow(M_PI,2);
  double Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;

  return Dv;

}

/*----- ROUTINE: r_delta -----
INPUT: cosmology, halo mass, scale factor, halo overdensity
TASK: Computes comoving halo radius assuming the overdensity criteria
*/
double r_delta(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status){

  double rho_matter = ccl_rho_x(cosmo, 1., 1, 1, status);

  return pow(halomass*3.0/(4.0*M_PI*rho_matter*odelta),1.0/3.0);

}

// This checks to make sure all necessary halo mass function parameters have been set-up,
// as well as associated splines.
static void ccl_cosmology_compute_hmfparams(ccl_cosmology *cosmo, int *status)
{
  if(cosmo->computed_hmfparams)
    return;

  // declare parameter splines on case-by-case basis
  switch(cosmo->config.mass_function_method) {
  case ccl_tinker10:{
    double delta[9] = {200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0};
    double lgdelta[9];
    double alpha[9] = {0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327};
    double beta[9] = {0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702};
    double gamma[9] ={0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81};
    double phi[9] = {-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49};
    double eta[9] = {-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336};
    int nd = 9;
    int i;

    for(i=0; i<nd; i++) {
      lgdelta[i] = log10(delta[i]);
    }

    gsl_spline * alphahmf = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(alphahmf, lgdelta, alpha, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating alpha(D) spline\n");
      return;
    }

    gsl_spline * betahmf  = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(betahmf, lgdelta, beta, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating beta(D) spline\n");
      return;
    }

    gsl_spline * gammahmf = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(gammahmf, lgdelta, gamma, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      gsl_spline_free(gammahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating gamma(D) spline\n");
      return;
    }

    gsl_spline * phihmf   = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(phihmf, lgdelta, phi, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      gsl_spline_free(gammahmf);
      gsl_spline_free(phihmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating phi(D) spline\n");
      return;
    }

    gsl_spline * etahmf   = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(etahmf, lgdelta, eta, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      gsl_spline_free(gammahmf);
      gsl_spline_free(phihmf);
      gsl_spline_free(etahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating eta(D) spline\n");
      return;
    }
    cosmo->data.alphahmf = alphahmf;
    cosmo->data.betahmf = betahmf;
    cosmo->data.gammahmf = gammahmf;
    cosmo->data.phihmf = phihmf;
    cosmo->data.etahmf = etahmf;
    cosmo->computed_hmfparams = true;

    break;
  }
  case ccl_tinker:{
    double delta[9] = {200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0};
    double lgdelta[9];
    double alpha[9] = {0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260};
    double beta[9] = {1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66};
    double gamma[9] ={2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41};
    double phi[9] = {1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44};
    int nd = 9;
    int i;

    for(i=0; i<nd; i++) {
      lgdelta[i] = log10(delta[i]);
    }

    gsl_spline * alphahmf = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(alphahmf, lgdelta, alpha, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating alpha(D) spline\n");
      return;
    }

    gsl_spline * betahmf  = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(betahmf, lgdelta, beta, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating beta(D) spline\n");
      return;
    }

    gsl_spline * gammahmf = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(gammahmf, lgdelta, gamma, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      gsl_spline_free(gammahmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating gamma(D) spline\n");
      return;
    }

    gsl_spline * phihmf   = gsl_spline_alloc(cosmo->spline_params.D_SPLINE_TYPE, nd);
    *status = gsl_spline_init(phihmf, lgdelta, phi, nd);
    if (*status) {
      gsl_spline_free(alphahmf);
      gsl_spline_free(betahmf);
      gsl_spline_free(gammahmf);
      gsl_spline_free(phihmf);
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_hmfparams(): Error creating phi(D) spline\n");
      return;
    }

    cosmo->data.alphahmf = alphahmf;
    cosmo->data.betahmf = betahmf;
    cosmo->data.gammahmf = gammahmf;
    cosmo->data.phihmf = phihmf;
    cosmo->computed_hmfparams = true;

    break;
  }
  default:
    // Error message could go here if we decide to make this public facing.
    // Currently not accessible from the API though.
    break;
  }
}

//TODO: some of these are unused, many are included in ccl.h

/*----- ROUTINE: ccl_massfunc_f -----
INPUT: cosmology+parameters, a halo mass, and scale factor
TASK: Outputs fitting function for use in halo mass function calculation;
  currently only supports:
    ccl_tinker (arxiv 0803.2706 )
    ccl_tinker10 (arxiv 1001.3162 )
    ccl_angulo (arxiv 1203.3216 )
    ccl_watson (arxiv 1212.0095 )
    ccl_shethtormen (arxiv 9901122)
*/
static double massfunc_f(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status)
{
  double fit_A, fit_a, fit_b, fit_c, fit_d, fit_p, overdensity_delta;
  double Omega_m_a;
  double delta_c_Tinker, nu;

  double sigma=ccl_sigmaM(cosmo, halomass, a, status);
  int gslstatus;

  switch(cosmo->config.mass_function_method) {

  // Equation (10) in arxiv: 9901122
  // Note that Sheth & Tormen (1999) use nu=(dc/sigma)^2 whereas we use nu=dc/sigma
  case ccl_shethtormen:

    // Check if odelta is outside the interpolated range
    if (odelta != Dv_BryanNorman(cosmo, a, status)) {
      *status = CCL_ERROR_HMF_DV;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: massfunc_f(): Sheth-Tormen called with not virial Delta_v\n");
      return NAN;
    }

    // ST mass function fitting parameters
    fit_A = 0.21616;
    fit_p = 0.3;
    fit_a = 0.707;

    // nu = delta_c(z) / sigma(M)
    nu = dc_NakamuraSuto(cosmo, a, status)/ccl_sigmaM(cosmo, halomass, a, status);

    return nu*fit_A*(1.+pow(fit_a*pow(nu,2),-fit_p))*exp(-fit_a*pow(nu,2)/2.);

  case ccl_tinker:

    // Check if odelta is outside the interpolated range
    if ((odelta < 200) || (odelta > 3200)) {
      *status = CCL_ERROR_HMF_INTERP;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: massfunc_f(): Tinker 2008 only supported in range of Delta = 200 to Delta = 3200.\n");
      return NAN;
    }

    // Compute HMF parameter (alpha, beta, gamma, phi) splines if they haven't
    // been computed already
    if (!cosmo->computed_hmfparams) {
      ccl_cosmology_compute_hmfparams(cosmo, status);
      ccl_check_status(cosmo, status);
    }
    gslstatus = gsl_spline_eval_e(cosmo->data.alphahmf, log10(odelta), NULL, &fit_A);
    gslstatus |= gsl_spline_eval_e(cosmo->data.betahmf, log10(odelta), NULL, &fit_a);
    gslstatus |= gsl_spline_eval_e(cosmo->data.gammahmf, log10(odelta), NULL, &fit_b);
    gslstatus |= gsl_spline_eval_e(cosmo->data.phihmf, log10(odelta), NULL, &fit_c);
    fit_d = pow(10, -1.0*pow(0.75 / log10(odelta / 75.0), 1.2));

    fit_A = fit_A*pow(a, 0.14);
    fit_a = fit_a*pow(a, 0.06);
    fit_b = fit_b*pow(a, fit_d);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_massfunc_f():");
      *status |= gslstatus;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_massfunc_f(): interpolation error for Tinker MF\n");
      return NAN;
    }
    return fit_A*(pow(sigma/fit_b,-fit_a)+1.0)*exp(-fit_c/sigma/sigma);
    break;
  case ccl_tinker10:
    // this version uses f(nu) parameterization from Eq. 8 in Tinker et al. 2010
    // use this for consistency with Tinker et al. 2010 fitting function for halo bias

    // Check if odelta is outside the interpolated range
    if ((odelta < 200) || (odelta > 3200)) {
      *status = CCL_ERROR_HMF_INTERP;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: massfunc_f(): Tinker 2010 only supported in range of Delta = 200 to Delta = 3200.\n");
      return 0;
    }

    if (!cosmo->computed_hmfparams) {
        ccl_cosmology_compute_hmfparams(cosmo, status);
        ccl_check_status(cosmo, status);
    }
    //critical collapse overdensity assumed in this model
    delta_c_Tinker = 1.686;
    nu = delta_c_Tinker/(sigma);

    gslstatus = gsl_spline_eval_e(cosmo->data.alphahmf, log10(odelta), NULL, &fit_A); //alpha in Eq. 8
    gslstatus |= gsl_spline_eval_e(cosmo->data.etahmf, log10(odelta), NULL, &fit_a); //eta in Eq. 8
    gslstatus |= gsl_spline_eval_e(cosmo->data.betahmf, log10(odelta), NULL, &fit_b); //beta in Eq. 8
    gslstatus |= gsl_spline_eval_e(cosmo->data.gammahmf, log10(odelta), NULL, &fit_c); //gamma in Eq. 8
    gslstatus |= gsl_spline_eval_e(cosmo->data.phihmf, log10(odelta), NULL, &fit_d); //phi in Eq. 8;

    fit_a *=pow(a, -0.27);
    fit_b *=pow(a, -0.20);
    fit_c *=pow(a, 0.01);
    fit_d *=pow(a, 0.08);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_massfunc_f():");
      *status |= gslstatus;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_massfunc_f(): interpolation error for Tinker 2010 MF\n");
      return NAN;
    }
    return nu*fit_A*(1.+pow(fit_b*nu,-2.*fit_d))*pow(nu, 2.*fit_a)*exp(-0.5*fit_c*nu*nu);
    break;

  case ccl_watson:
    if(odelta!=200.) {
      *status = CCL_ERROR_HMF_INTERP;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_massfunc_f(): Watson HMF only supported for Delta = 200.\n");
      return NAN;
    }
    // these parameters from: Angulo et al 2012 (arxiv 1203.3216 )
    Omega_m_a = ccl_omega_x(cosmo, a, ccl_species_m_label,status);
    fit_A = Omega_m_a*(0.990*pow(a,3.216)+0.074);
    fit_a = Omega_m_a*(5.907*pow(a,3.599)+2.344);
    fit_b = Omega_m_a*(3.136*pow(a,3.058)+2.349);
    fit_c = 1.318;

    return fit_A*(pow(sigma/fit_b,-fit_a)+1.0)*exp(-fit_c/sigma/sigma);

  case ccl_angulo:
    if(odelta!=200.) {
      *status = CCL_ERROR_HMF_INTERP;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_massfunc_f(): Angulo HMF only supported for Delta = 200.\n");
      return NAN;
    }
    // these parameters from: Watson et al 2012 (arxiv 1212.0095 )
    fit_A = 0.201;
    fit_a = 2.08;
    fit_b = 1.7;
    fit_c = 1.172;

    return fit_A*pow( (fit_a/sigma)+1.0, fit_b)*exp(-fit_c/sigma/sigma);

  default:
    *status = CCL_ERROR_MF;
    ccl_cosmology_set_status_message(cosmo ,
	    "ccl_massfunc.c: ccl_massfunc(): Unknown or non-implemented mass function method: %d \n",
	    cosmo->config.mass_function_method);
    return NAN;
  }
}

static double ccl_halo_b1(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status)
{
  double fit_A, fit_B, fit_C, fit_a, fit_b, fit_c, fit_p, overdensity_delta, y;
  double delta_c_Tinker, nu;
  double sigma=ccl_sigmaM(cosmo,halomass,a, status);
  switch(cosmo->config.mass_function_method) {

  // Equation (12) in  arXiv: 9901122
  // Derived using the peak-background split applied to the mass function in the same paper
  // Note that Sheth & Tormen (1999) use nu=(dc/sigma)^2 whereas we use nu=dc/sigma
  case ccl_shethtormen:

    // Check if Delta_v is the virial Delta_v
    if (odelta != Dv_BryanNorman(cosmo, a, status)) {
      *status = CCL_ERROR_HMF_DV;
      ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: halo_b1(): Sheth-Tormen called with not virial Delta_v\n");
      return NAN;
    }

    // ST bias fitting parameters (which are the same as for the mass function)
    fit_p = 0.3;
    fit_a = 0.707;

    // Cosmology dependent delta_c and nu
    double delta_c = dc_NakamuraSuto(cosmo, a, status);
    nu = delta_c/ccl_sigmaM(cosmo, halomass, a, status);

    return 1.+(fit_a*pow(nu,2)-1.+2.*fit_p/(1.+pow(fit_a*pow(nu,2),fit_p)))/delta_c;

    //this version uses b(nu) parameterization, Eq. 6 in Tinker et al. 2010
    // use this for consistency with Tinker et al. 2010 fitting function for halo bias
  case ccl_tinker10:
    y = log10(odelta);
    //critical collapse overdensity assumed in this model
    delta_c_Tinker = 1.686;
    //peak height - note that this factorization is incorrect for e.g. massive neutrino cosmologies
    nu = delta_c_Tinker/(sigma);
    // Table 2 in https://arxiv.org/pdf/1001.3162.pdf
    fit_A = 1.0 + 0.24*y*exp(-pow(4./y,4.));
    fit_a = 0.44*y-0.88;
    fit_B = 0.183;
    fit_b = 1.5;
    fit_C = 0.019+0.107*y+0.19*exp(-pow(4./y,4.));
    fit_c = 2.4;

    return 1.-fit_A*pow(nu,fit_a)/(pow(nu,fit_a)+pow(delta_c_Tinker,fit_a))+fit_B*pow(nu,fit_b)+fit_C*pow(nu,fit_c);
    break;

  default:
    *status = CCL_ERROR_MF;
    ccl_cosmology_set_status_message(cosmo ,
	    "ccl_massfunc.c: ccl_halo_b1(): No b(M) fitting function implemented for mass_function_method: %d \n",
      cosmo->config.mass_function_method);
    return 0;
  }
}

void ccl_cosmology_compute_sigma(ccl_cosmology *cosmo, int *status)
{
  if(cosmo->computed_sigma)
    return;

  // create linearly-spaced values of the mass.
  int nm=cosmo->spline_params.LOGM_SPLINE_NM;
  double * m = ccl_linear_spacing(cosmo->spline_params.LOGM_SPLINE_MIN, cosmo->spline_params.LOGM_SPLINE_MAX, nm);

  // create space for y, to be filled with sigma and dlnsigma_dlogm
  double * y = malloc(sizeof(double)*nm);
  double smooth_radius;
  double na, nb;

  // start up of GSL pointers
  int gslstatus = 0;
  gsl_spline *logsigma;
  gsl_spline *dlnsigma_dlogm;

  if (m==NULL ||
      (fabs(m[0]-cosmo->spline_params.LOGM_SPLINE_MIN)>1e-5) ||
      (fabs(m[nm-1]-cosmo->spline_params.LOGM_SPLINE_MAX)>1e-5) ||
      (m[nm-1]>10E17)
      ) {
    *status = CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo,"ccl_cosmology_compute_sigmas(): Error creating linear spacing in m\n");
  }

  // fill in sigma, if no errors have been triggered at this time.
  if (*status == 0) {
    for (int i=0; i<nm; i++) {
      smooth_radius = ccl_massfunc_m2r(cosmo, pow(10,m[i]), status);
      y[i] = log10(ccl_sigmaR(cosmo, smooth_radius, 1., status));
    }
    logsigma = gsl_spline_alloc(cosmo->spline_params.M_SPLINE_TYPE, nm);
    *status = gsl_spline_init(logsigma, m, y, nm);
  }

  if (*status !=0 ) {
    *status = CCL_ERROR_SPLINE ;
    ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating sigma(M) spline\n");
  }

  // again, making splines assuming nothing bad has happened to this point

  if (*status == 0 ) {
    for (int i=0; i<nm; i++) {
      if(i==0) {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i], NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]+cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&nb);
        y[i] = (na-nb)*log(10.);
        y[i] = 2.*y[i] / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
      else if (i==nm-1) {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]-cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i], NULL,&nb);
        y[i] = (na-nb)*log(10.);
        y[i] = 2.*y[i] / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
      else {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]-cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]+cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&nb);
        y[i] = (na-nb)*log(10.);
        y[i] = y[i] / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
    }
  }

  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_cosmology_compute_sigma():");
    *status |= gslstatus;
    *status = CCL_ERROR_SPLINE ;
    ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error evaluating grid points for dlnsigma/dlogM spline\n");
  }

  if(*status==0) {
    dlnsigma_dlogm = gsl_spline_alloc(cosmo->spline_params.M_SPLINE_TYPE, nm);
    *status = gsl_spline_init(dlnsigma_dlogm, m, y, nm);
  }

  if(*status!=0) {
    *status = CCL_ERROR_SPLINE ;
    ccl_cosmology_set_status_message(cosmo, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating dlnsigma/dlogM spline\n");
  }


  cosmo->data.logsigma = logsigma;
  cosmo->data.dlnsigma_dlogm = dlnsigma_dlogm;
  cosmo->computed_sigma = true;

  free(m);
  free(y);
  if(*status != 0) {
    gsl_spline_free(logsigma);
    gsl_spline_free(dlnsigma_dlogm);
  }
  return;
}

/*----- ROUTINE: ccl_dlninvsig_dlogm -----
INPUT: ccl_cosmology *cosmo, double halo mass in units of Msun
TASK: returns the value of the derivative of ln(sigma^-1) with respect to log10 in halo mass.
*/

static double ccl_dlninvsig_dlogm(ccl_cosmology *cosmo, double halomass, int*status)
{

  if (cosmo->params.N_nu_mass>0){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Support for the halo mass function in cosmologies with massive neutrinos is not yet implemented.\n");
	  return NAN; 
  }
  
  // Raise an error if we have mu / Sigma modifcation to gravity turned on
  if (fabs(cosmo->params.mu_0)>1e-14 || fabs(cosmo->params.sigma_0)>1e-14){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power(): The mass function is not implemented the mu / Sigma modified gravity parameterisation.\n");
	  return NAN;
  }	

  if (!cosmo->computed_sigma) {
    ccl_cosmology_compute_sigma(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  double val, logmass;

  logmass = log10(halomass);

  int gslstatus = gsl_spline_eval_e(cosmo->data.dlnsigma_dlogm, logmass, NULL, &val);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_massfunc():");
    *status |= gslstatus;
  }
  ccl_check_status(cosmo, status);

  return val;
}

/*----- ROUTINE: ccl_massfunc -----
INPUT: ccl_cosmology * cosmo, double halo mass in units of Msun, double scale factor
TASK: returns halo mass function as dn/dlog10(m) in comoving Msun^-1 Mpc^-3 (haloes per mass interval per volume)
*/
double ccl_massfunc(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status)
{
  if (cosmo->params.N_nu_mass>0){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Support for the halo mass function in cosmologies with massive neutrinos is not yet implemented.\n");
	  return NAN;
  }
  
  if (fabs(cosmo->params.mu_0)>1e-14 || fabs(cosmo->params.sigma_0)>1e-14){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  strcpy(cosmo->status_message,"ccl_massfunc.c: ccl_massfunc(): The halo mass funcion is not implemented the mu / Sigma modified gravity parameterisation.\n");
	  return NAN;
  }

  double f, rho_m;

  rho_m = ccl_constants.RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  f=massfunc_f(cosmo,halomass,a,odelta,status);

  return f*rho_m*ccl_dlninvsig_dlogm(cosmo,halomass,status)/halomass;
}

/*----- ROUTINE: ccl_halob1 -----
INPUT: ccl_cosmology * cosmo, double halo mass in units of Msun, double scale factor
TASK: returns dimensionless linear halo bias
*/
double ccl_halo_bias(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status)
{
  if (cosmo->params.N_nu_mass>0){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Support for the halo bias in cosmologies with massive neutrinos is not yet implemented.\n");
	  return NAN;
  }
  
  if (fabs(cosmo->params.mu_0)>1e-14 || fabs(cosmo->params.sigma_0)>1e-14){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  strcpy(cosmo->status_message,"ccl_massfunc.c: ccl_halobias(): The halo bias is not implemented the mu / Sigma modified gravity parameterisation.\n");
	  return NAN;
  }	
		
  if (!cosmo->computed_sigma) {
    ccl_cosmology_compute_sigma(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  double f;
  f = ccl_halo_b1(cosmo,halomass,a,odelta, status);
  ccl_check_status(cosmo, status);
  return f;
}
/*---- ROUTINE: ccl_massfunc_m2r -----
INPUT: ccl_cosmology * cosmo, halomass in units of Msun
TASK: takes halo mass and converts to halo radius
  in units of Mpc.
*/
double ccl_massfunc_m2r(ccl_cosmology *cosmo, double halomass, int *status)
{
  double rho_m, smooth_radius;

  // Comoving matter density
  //rho_m = ccl_constants.RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  rho_m = ccl_rho_x(cosmo, 1., ccl_species_m_label, 1, status);

  smooth_radius = pow((3.0*halomass) / (4*M_PI*rho_m), (1.0/3.0));

  return smooth_radius;
}

/*----- ROUTINE: ccl_sigma_M -----
INPUT: ccl_cosmology * cosmo, double halo mass in units of Msun, double scale factor
TASK: returns sigma from the sigmaM interpolation. Also computes the sigma interpolation if
necessary.
*/
double ccl_sigmaM(ccl_cosmology *cosmo, double halomass, double a, int *status)
{
  if (cosmo->params.N_nu_mass>0){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Support for the sigma(M) function in cosmologies with massive neutrinos is not yet implemented.\n");
	  return NAN;
  }

  double sigmaM;
  // Check if sigma has already been calculated
  if (!cosmo->computed_sigma) {
    ccl_cosmology_compute_sigma(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  double lgsigmaM;

  int gslstatus = gsl_spline_eval_e(cosmo->data.logsigma, log10(halomass), NULL, &lgsigmaM);

  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_sigmaM():");
    *status |= gslstatus;
  }

  // Interpolate to get sigma
  sigmaM = pow(10,lgsigmaM)*ccl_growth_factor(cosmo, a, status);
  ccl_check_status(cosmo, status);
  return sigmaM;
}
