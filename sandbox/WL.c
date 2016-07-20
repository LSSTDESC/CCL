#include "gsl/gsl_spline.h"
#include "math.h"

// from one of the headers
#define A_SPLINE_TYPE gsl_interp_akima


void ccl_compute_2d_power_spectra(int n_tomographic_bin, double zmax, gsl_spline * photoz_splines[],
    // Some kind of 2D spline for P(k) needed also?
    int * status
    ){
    for (int i=0; i<n_tomographic_bin; i++){
        // Get the specific spline for this tomographic bin
        gsl_spline * photoz_spline = photoz_splines[i];

        // Do the calculation you want with this spline.
        // Or maybe you want another tomographic bin to do cross-correlations.
    }
}



// Calling function example
int ccl_example_projected_2d_example(){
#define NTOMO_EXAMPLE 4
    gsl_spline * photoz_splines[NTOMO_EXAMPLE];
    double zmax;
    // Generate Gaussian photo-zs just as an example
    for (int i=0; i<NTOMO_EXAMPLE; i++){
            int nz = 200;

            gsl_spline * photoz_spline = gsl_spline_alloc(A_SPLINE_TYPE, nz);
            double z[nz];
            double n_of_z[nz];

            // Build the array of z, n(z)
            for (int j=0; j<nz; j++){
                z[j] = 0.01*j;
                n_of_z[j] = exp(-0.5*pow((z[j]-(i+1)*0.2)/0.1,2));
            }

            // Init spline; should check error.
            gsl_spline_init(photoz_spline, z, n_of_z, nz);

            // Fill in array of splines with this new one.
            photoz_splines[i] = photoz_spline;

            // Might also need this later
            zmax = n_of_z[nz-1];
    }

    // Call the function
    int status=0;
    ccl_compute_2d_power_spectra(NTOMO_EXAMPLE, zmax, photoz_splines, &status);

    return 0;
}