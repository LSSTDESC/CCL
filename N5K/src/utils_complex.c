#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "utils.h"

double complex gamma_lanczos(double complex z) {
/* Lanczos coefficients for g = 7 */
	static double p[] = {
		0.99999999999980993227684700473478,
		676.520368121885098567009190444019,
		-1259.13921672240287047156078755283,
		771.3234287776530788486528258894,
		-176.61502916214059906584551354,
		12.507343278686904814458936853,
		-0.13857109526572011689554707,
		9.984369578019570859563e-6,
		1.50563273514931155834e-7};

	if(creal(z) < 0.5) {return M_PI / (csin(M_PI*z)*gamma_lanczos(1. - z));}
	z -= 1;
	double complex x = p[0];
	for(int n = 1; n < 9; n++){ x += p[n] / (z + (double)(n));}

	double complex t = z + 7.5;
	return sqrt(2*M_PI) * cpow(t, z+0.5) * cexp(-t) * x;
}

double complex lngamma_lanczos(double complex z) {
/* Lanczos coefficients for g = 7 */
	static double p[] = {
		0.99999999999980993227684700473478,
		676.520368121885098567009190444019,
		-1259.13921672240287047156078755283,
		771.3234287776530788486528258894,
		-176.61502916214059906584551354,
		12.507343278686904814458936853,
		-0.13857109526572011689554707,
		9.984369578019570859563e-6,
		1.50563273514931155834e-7};

	if(creal(z) < 0.5) {return clog(M_PI) - clog(csin(M_PI*z)) - lngamma_lanczos(1. - z);}
	z -= 1;
	double complex x = p[0];
	for(int n = 1; n < 9; n++){ x += p[n] / (z + (double)(n));}

	double complex t = z + 7.5;
	return log(2*M_PI) /2.  + (z+0.5)*clog(t) -t + clog(x);
}

double complex ln_g_m_vals(double mu, double complex q) {
/* similar routine as python version.
use asymptotic expansion for large |mu+q| */
	double complex asym_plus = (mu+1+ q)/2.;
	double complex asym_minus= (mu+1- q)/2.;

	return (asym_plus-0.5)*clog(asym_plus) - (asym_minus-0.5)*clog(asym_minus) - q \
		+1./12 *(1./asym_plus - 1./asym_minus) \
		+1./360.*(1./cpow(asym_minus,3) - 1./cpow(asym_plus,3)) \
		+1./1260*(1./cpow(asym_plus,5) - 1./cpow(asym_minus,5));
}

void g_l(double l, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
Calculate g_l = exp( zln2 + lngamma( (l+nu)/2 + I*eta/2 ) - lngamma( (3+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			// gl[i] = cexp(z*log(2.) + clog(gamma_lanczos((l+z)/2.) ) - clog(gamma_lanczos((3.+l-z)/2.)));
			gl[i] = cexp(z*log(2.) + lngamma_lanczos((l+z)/2.) - lngamma_lanczos((3.+l-z)/2.) );	
		}else{
			gl[i] = cexp(z*log(2.) + ln_g_m_vals(l+0.5, z-1.5));
		}
		//printf("i: %ld, eta: %f, real: %f, imag: %f\n", i, eta[i], creal(cexp(z*log(2.) + lngamma_lanczos((l+z)/2.) - lngamma_lanczos((3.+l-z)/2.) ))/creal(cexp(z*log(2.) + ln_g_m_vals(l+0.5, z-1.5))), cimag(cexp(z*log(2.) + lngamma_lanczos((l+z)/2.) - lngamma_lanczos((3.+l-z)/2.) ))/cimag(cexp(z*log(2.) + ln_g_m_vals(l+0.5, z-1.5))));
		//printf("i: %ld, eta: %f, real: %f, imag: %f\n", i, eta[i], creal(gl[i]), cimag(gl[i]));

		// if(isnan(gl[i])) {printf("nan at l,nu,eta, = %lf %lg %lg %lg %lg\n", l,nu, eta[i], gamma_lanczos((l+z)/2.),gamma_lanczos((3.+l-z)/2.));exit(0);}
	}
}

void g_l_1(double l, double nu, double *eta, double complex *gl1, long N) {
/* z = nu + I*eta
Calculate g_l_1 = exp(zln2 + lngamma( (l+nu-1)/2 + I*eta/2 ) - lngamma( (4+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + lngamma_lanczos((l+z-1.)/2.) - lngamma_lanczos((4.+l-z)/2.));
		}else{
			gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + ln_g_m_vals(l+0.5, z-2.5));
		}
	}
}

void g_l_2(double l, double nu, double *eta, double complex *gl2, long N) {
/* z = nu + I*eta
Calculate g_l_1 = exp(zln2 + lngamma( (l+nu-2)/2 + I*eta/2 ) - lngamma( (5+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			gl2[i] = (z-1.)* (z-2.)* cexp((z-2.)*log(2.) + lngamma_lanczos((l+z-2.)/2.) - lngamma_lanczos((5.+l-z)/2.));
		}else{
			gl2[i] = (z-1.)* (z-2.)* cexp((z-2.)*log(2.) + ln_g_m_vals(l+0.5, z-3.5));
		}
	}
}

void g_l_modified(double l, double nu, double *eta, double complex *gl, long N) {
	//long i;
	//double complex z;
	g_l(l, nu-2, eta, gl, N);
	//for(i=0; i<N; i++) {
	//	z = nu+I*eta[i];
	//	gl[i] /= ((l-2.+z)*(3.+l-z));
	//}
}
void g_l_1_modified(double l, double nu, double *eta, double complex *gl, long N) {
	//long i;
	//double complex z;
	//g_l(l, nu, eta, gl, N);
	//for(i=0; i<N; i++) {
	//	z = nu+I*eta[i];
	//	gl[i] /= ((l-2.+z)*(3.+l-z));
	//}	
	g_l_1(l, nu-2, eta, gl, N);
	//for(i=0; i<N; i++) {
	//	z = nu+I*eta[i];
	//	gl[i] /= ((l-2.+z)*(3.+l-z));
	//}
}
void g_l_2_modified(double l, double nu, double *eta, double complex *gl, long N) {
	//long i;
	//double complex z;
	g_l_2(l, nu-2, eta, gl, N);
	//for(i=0; i<N; i++) {
	//	z = nu+I*eta[i];
	//	gl[i] /= ((l-2.+z)*(3.+l-z));
	//}
}

void c_window(double complex *out, double c_window_width, long halfN) {
	// 'out' is (halfN+1) complex array
	long Ncut;
	Ncut = (long)(halfN * c_window_width);
	long i;
	double W;
	for(i=0; i<=Ncut; i++) { // window for right-side
		W = (double)(i)/Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI/Ncut);
		out[halfN-i] *= W;
		    //printf("%ld %f %f %f \n", i, W, creal(out[halfN-i]), cimag(out[halfN-i]));

	}
}

