#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"

void ccl_cosmology_compute_power_bbks(ccl_cosmology * cosmo, int *status){

    if (*status){
        return;
    }


    double kmin = K_MIN;
    double kmax = K_MAX;
    int nk = N_K;

    // The x array is initially k, but will later
    // be overwritten with log(k)
    double * x = ccl_log_spacing(kmin, kmax, nk);
    double * y = malloc(sizeof(double)*nk);

    if (y==NULL|| x==NULL){
        fprintf(stderr, "Could not allocate memory for power\n");
        free(x);
        free(y);
        *status = 1;
        return;
    }

    // After this loop k will contain 
    for (int i=0; i<nk; i++){
        y[i] = log(ccl_bbks_power(&cosmo->params, x[i]));
        x[i] = log(x[i]);
    }

    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_lin, x, y, nk);

    double sigma_8 = ccl_sigma8(log_power_lin, cosmo->params.h, status);
    double log_sigma_8 = log(cosmo->params.sigma_8) - log(sigma_8);
    for (int i=0; i<nk; i++){
        y[i] += log_sigma_8;
    }

    gsl_spline_free(log_power_lin);
    log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_lin, x, y, nk);    



    gsl_spline * log_power_nl = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_nl, x, y, nk);

    free(x);
    free(y);

    cosmo->data.p_lin = log_power_lin;
    cosmo->data.p_nl = log_power_nl;


}




void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int *status){
    //either:
    // if (cosmo->computed_distances) return; // if immutable
    //or
    // if (!ccl_needs_recomputation(cosmo)) return; // if immutable

    ccl_cosmology_compute_distances(cosmo, status);

    if (*status){
        return;
    }

    switch(cosmo->config.transfer_function_method){
        case bbks:
            ccl_cosmology_compute_power_bbks(cosmo, status);
            break;

        default:
            fprintf(stderr, "Unknown or non-implemented transfer function method\n");
            *status =1;
            return;
    }
    return;

}


double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k, int * status){
    ccl_cosmology_compute_power(cosmo, status);
    if (*status) return NAN;

    // log power at a=1 (z=0)
    double log_p_1 = gsl_spline_eval(cosmo->data.p_lin, log(k), NULL);
    // TODO: GSL spline error handling

    double p_1 = exp(log_p_1);

    if (a==1){
        return p_1;
    }

    double D = ccl_growth_factor(cosmo, a, status);
    double p = D*D*p_1;
    if (*status){
        p = NAN;
    }
    return p;
}

