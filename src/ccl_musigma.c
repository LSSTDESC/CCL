#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


/* --------- ROUTINE: ccl_mu_MG ---------
INPUT: cosmology object, scale factor, wavenumber for scale
TASK: Compute mu(a,k) where mu is one of the the parameterizating functions
of modifications to GR in the quasistatic approximation.
*/

double ccl_mu_MG(ccl_cosmology * cosmo, double a, double k, int *status)
{
    double s1_k, s2_k, hnorm;
	// This function can be extended to include other
	// redshift and scale z-dependences for mu in the future
    if (k==0.0) {
        s1_k = cosmo->params.c1_mg;
    }
    else {
      hnorm = ccl_h_over_h0(cosmo, a, status);
	    s2_k = (cosmo->params.lambda_mg*(hnorm*cosmo->params.H0)/k/(ccl_constants.CLIGHT/1000));
	    s1_k = (1.0+cosmo->params.c1_mg*s2_k*s2_k)/(1.0+s2_k*s2_k);
	}
	return cosmo->params.mu_0 * ccl_omega_x(cosmo, a, ccl_species_l_label, status)/cosmo->params.Omega_l*s1_k;
}

/* --------- ROUTINE: ccl_Sig_MG ---------
INPUT: cosmology object, scale factor, wavenumber for scale
TASK: Compute Sigma(a,k) where Sigma is one of the the parameterizating functions
of modifications to GR in the quasistatic approximation.
*/

double ccl_Sig_MG(ccl_cosmology * cosmo, double a, double k, int *status)
{
    double s1_k, s2_k, hnorm;
	// This function can be extended to include other
	// redshift and scale dependences for Sigma in the future.
    if (k==0.0) {
        s1_k = cosmo->params.c2_mg;
    }
    else {
      hnorm = ccl_h_over_h0(cosmo, a, status);
	    s2_k = cosmo->params.lambda_mg*(hnorm*cosmo->params.H0)/k/(ccl_constants.CLIGHT/1000);
        s1_k = (1.0+cosmo->params.c2_mg*s2_k*s2_k)/(1.0+s2_k*s2_k);

	}
	return cosmo->params.sigma_0 * ccl_omega_x(cosmo, a, ccl_species_l_label, status)/cosmo->params.Omega_l*s1_k;
}
