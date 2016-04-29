#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "ccl_core.h"

/*
a) Growth factor at z = 0,1,2,3,4,5
b) Comoving radial distance [Mpc/h] at z = 0,1,2,3,4,5
c) Linear matter power spectrum (from BBKS) [(Mpc/h)^3] at z = 0,1,2,3; 1.e-3 h/Mpc <= k <= 10, 10 bins/decade
d)\sigma(M) for M = {1.e+6, 1.e+8, 1.e+10, 1.e+12, 1.e+14, 1.e+16} M_sun/h (this replaces n(M), b(M) from my previous email - if one can compute \sigma, implementing any of these fitting functions should be straight forward)

with a relative accuracy goal of 10.e-4. Would two weeks be a reasonable time scale for a first iteration on step 1?

In the next step, we will specify a redshift distribution (analytically, and as a histogram) and compare angular power spectra. The final step will be angular correlation functions.

Cosmological Models:
Model1 (flat LCMD): Omega_m = 0.3, Omega_b = 0.05, Omega_v = 0.7, h0 = 0.7, sigma_8 = 0.8, n_s = 0.96, w_0 = -1.0, w_a = 0.0

Model2 (w0 LCMD): Omega_m = 0.3, Omega_b = 0.05, Omega_v = 0.7, h0 = 0.7, sigma_8 = 0.8, n_s = 0.96, w_0 = -0.9, w_a = 0.0

Model3 (wa LCMD): Omega_m = 0.3, Omega_b = 0.05, Omega_v = 0.7, h0 = 0.7, sigma_8 = 0.8, n_s = 0.96, w_0 = -0.9, w_a = 0.1
*/

/* 
	Error handling models:
	- FITSIO style - pointer to status
	- Standard - return an integer status
	- gsl style - customizable error handler; default prints out an error and quits program.
*/

/*
	TODO List

	- mailing list
	- decide error model
	- 
*/




/* 
	This is a utility function to compare two arrays of numbers and 
	see if the values in them are the same to some requested accuracy level.
	It prints out the results.


*/
int check_fractional_errors(int n, double values[n], double test[n], double ftol, const char * message){
	int status = 0;
	for (int i=0; i<n; i++){
		double frac_error;

		// We would like to test zero values, but cannot do those
		// with fraction error.  On the assumption that if the value
		// is zero we should require near-machine precision, we test 
		// as if the target value is 1.0 instead;
		if (test[i]==0){
			frac_error = values[i];
		}
		else{
			frac_error = values[i]/test[i]-1.0;
		}

		// Print an error message on failure, one for each failing element
		if (!(fabs(frac_error) < ftol)){
			printf("FAIL  Fractional error %le>%le at i=%d: %s\n", frac_error, ftol, i, message);
			status = 1;
		}
	}
	if (status==0) printf("SUCCESS %s\n", message);

	return status;
}



/*
	Compare the results of a function of 1 variable to those from a test sample.

	We will use this function to test the various distance functions.

*/


int test_1d_functions(
	ccl_cosmology * cosmo, 
	const int na, double a_samples[na], double target_values[na], double ftol,
	double (*scalar_function)(ccl_cosmology *, double),
	int (*array_function)(ccl_cosmology *, int, double*, double*),
	const char * test_name
	){

	int status = 0;

	// Initialize the test_values to nan so the test cannot fail by accident 
	// (e.g. if everything is zero)
	double test_values[na];
	for (int i=0; i<na; i++) test_values[i] = NAN;

	// Test the scalar form of the function
	for (int i=0; i<na; i++) test_values[i] = scalar_function(cosmo, a_samples[i]);
	status |= check_fractional_errors(na, test_values, target_values, ftol, test_name);

	// Reset the test values; otherwise this function could do nothing
	// and still pass
	for (int i=0; i<na; i++) test_values[i] = NAN;
	
	// Now check the vector form.
	status |= array_function(cosmo, na, a_samples, test_values);
	if (status){
			printf("FAIL  Array function for %s reported that it failed\n", test_name);
	}
	status |= check_fractional_errors(na, test_values, target_values, ftol, test_name);

	return status;
}

/*
	Compare the results of a 2D function to a set of test values.

	We will use this for linear matter power and perhaps sigma later.
	We tell it the grid values and the expected values on those grid points.
*/

int test_2d_functions(
	ccl_cosmology * cosmo, 
	const int na, const int nk, double a_samples[na], double k_samples[nk], double target_values[na*nk], double ftol,
	double (*scalar_function)(ccl_cosmology *, double, double),
	int (*array_function)(ccl_cosmology *, int, double *, double *, double*),
	const char * test_name
	){

	int status = 0;

	// "t" is for total - total number of grid points
	int nt = na*nk;

	double test_values[nt];
	for (int i=0; i<nt; i++) test_values[i] = NAN;

	// Loop through the k and array arrays (2D)
	// and build up the grid of test points.
	int it = 0;
	for (int ik=0; ik<nk; ik++){
		for (int ia=0; ia<na; ia++){
			double k = k_samples[ik];
			double a = a_samples[ia];
			test_values[it] = scalar_function(cosmo, a, k);
			it++;
		}
	}

	// Compare arrays
	status |= check_fractional_errors(na, test_values, target_values, ftol, test_name);



	// Reset test values
	for (int i=0; i<nt; i++) test_values[i] = NAN;

	// Now test the array form.
	// We first turn the k and a arrays into two longer nk*na arrays
	// so that we have the entire set of grid points we are testing on.
	double test_k[nt];
	double test_a[nt];
	it = 0;
	for (int ik=0; ik<nk; ik++){
		for (int ia=0; ia<na; ia++){
			test_k[it] = k_samples[ik];
			test_a[it] = a_samples[ia];
			it++;
		}
	}

	// Run array calc
	// There may be a better form to put this in. This assumes just pairs of a and k.
	// Whereas we may want e.g. slices at fixed a.
	test_values[it++] = array_function(cosmo, nt, test_a, test_k, test_values);

	// Compare arrays
	status |= check_fractional_errors(na, test_values, target_values, ftol, test_name);



	return status;
}




int test_lcdm(){
	
	// The cosmological parameters we will use in this test
	double Omega_c = 0.25;
	double Omega_b = 0.05;
	double h0 = 0.7;
	double sigma_8 = 0.8;
	double n_s = 0.96;

	// Unused; the create_lcdm function does not require them
	// double w0 = -1.0;
	// double wa = 0.0;


	// Create the cosmology
	int status = 0;
	int i;	
	ccl_cosmology * lcdm = NULL;
	lcdm = ccl_create_lcdm(Omega_c, Omega_b, h0, sigma_8, n_s);

	// Target points for the 1D tests
	const int nz = 6;
	double test_z[nz] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	double test_a[nz];
	for (int i=0; i<nz; i++) test_a[i] = 1.0/(1+test_z[i]);


	// Target values for the 1D tests.
	// TODO: Fill these in.
	double expected_comoving[nz] = {NAN, NAN, NAN, NAN, NAN, NAN};
	double ftol_comoving = 1e-4;

	double expected_growth[nz] = {NAN, NAN, NAN, NAN, NAN, NAN};
	double ftol_growth = 1e-4;


	// Run the tests on the 1D functions.
	// Can add more functions here.
	status |= test_1d_functions(lcdm, 
		nz, test_a, expected_comoving, ftol_comoving, 
		ccl_comoving_radial_distance, ccl_comoving_radial_distances,
		"Comoving radial distance"
		);

	status |= test_1d_functions(lcdm, 
		nz, test_a, expected_growth, ftol_growth, 
		ccl_growth_factor, ccl_growth_factors,
		"Growth factor"
		);



	// Sample values for the 2D P(k,a) tests.
	// First in z (or a)
	const int nz_power = 4;
	double power_z[nz_power] = {0.0,1.0,2.0,3.0};
	double power_a[nz_power];
	for (i=0; i<nz_power; i++) power_a[i] = 1.0/(1+power_z[i]);



	// Now the sample points in k
	// c) Linear matter power spectrum (from BBKS) [(Mpc/h)^3] at z = 0,1,2,3; 1.e-3 h/Mpc <= k <= 10, 10 bins/decade	
	const int nk_power = 41;
	double power_k[nk_power];
	for (i=0; i<nk_power; i++) power_k[i] = 1e-3 * pow(1.2589254117941675, i);

	// Check I didn't screw up the number in the line above.
	assert(fabs(power_k[nk_power-1]-10.0)<1e-10);


	const int nt_power = nk_power * nz_power;
	double expected_linear_power[nt_power];


	// Until we have the required values, just fill this in 
	// with NANs so it will never succeed.
	for (i=0; i<nt_power; i++) expected_linear_power[i] = NAN;

	

	// Run the test.
	status |= test_2d_functions(
		lcdm, 
		nz_power, nk_power, power_a, power_k, expected_linear_power, 1e-4,
		ccl_linear_matter_power, ccl_linear_matter_powers,
		"Linear matter power"
	);



	ccl_free(lcdm);

	if (status){
		printf("\n TESTS FAILED :-(\n\n");
	}
	else{
		printf("\n TESTS PASSED\n\n");
	}

	// ccl_linear_matter_power(lcdm, k, a);
	return status;


}

int main(){
	int status = 0;
	status |= test_lcdm();
	return status;
}