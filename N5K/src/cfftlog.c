#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include <time.h>

#include <fftw3.h>

#include "utils.h"
#include "utils_complex.h"
#include "cfftlog.h"

void cfftlog_wrapper(double *x, double *fx, long N, double ell, double *y, double *Fy, double nu, double c_window_width, int derivative, long N_pad){
	config my_config;
	my_config.nu = nu;
	my_config.c_window_width = c_window_width;
	my_config.derivative = derivative;
	my_config.N_pad = N_pad;
	cfftlog(x, fx, N, &my_config, ell, y, Fy);
}

void cfftlog(double *x, double *fx, long N, config *config, double ell, double *y, double *Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	switch(config->derivative) {
		case 0: g_l(ell, config->nu, eta_m, gl, halfN+1); break;
		case 1: g_l_1(ell, config->nu, eta_m, gl, halfN+1); break;
		case 2: g_l_2(ell, config->nu, eta_m, gl, halfN+1); break;
		default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
	}
	// printf("g2[0]: %.15e+I*(%.15e)\n", creal(g2[0]),cimag(g2[0]));

	// calculate y arrays
	for(i=0; i<N_original; i++) {y[i] = (ell+1.) / x[N_original-1-i];}
	y0 = y[0];

	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);

	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	for(i=0; i<=halfN; i++) {
		//if (config->derivative==1) printf("i: %ld, eta: %f, real: %f, imag: %f, real: %f, imag: %f\n", i, eta_m[i], creal(out[i]), cimag(out[i]),creal(gl[i]), cimag(gl[i]));

		out[i] *= gl[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]);
		out[i] = conj(out[i]);
		//if (config->derivative==0) printf("i: %ld, eta: %f, base: %f, real: %f, imag: %f\n",i, eta_m[i],  creal(cpow(0.0, -I*eta_m[i])), creal(cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i])), cimag(cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i])));
		//if (config->derivative==0) printf("i: %ld, eta: %f, real: %f, imag: %f\n", i, eta_m[i], creal(out[i]), cimag(out[i]));
	}

	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));
	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out, out_ifft, FFTW_ESTIMATE);

	fftw_execute(plan_backward);

	for(i=0; i<N_original; i++) {
		Fy[i] = out_ifft[i-N_pad] * sqrt(M_PI) / (4.*N * pow(y[i], config->nu));
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	free(out_ifft);
}

void cfftlog_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, long N_pad){
	config my_config;
	my_config.nu = nu;
	my_config.c_window_width = c_window_width;
	my_config.derivative = derivative;
	my_config.N_pad = N_pad;
	cfftlog_ells(x, fx, N, &my_config, ell, Nell, y, Fy);
}

void cfftlog_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		switch(config->derivative) {
			case 0: g_l(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 1: g_l_1(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 2: g_l_2(ell[j], config->nu, eta_m, gl, halfN+1); break;
			default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
		}

		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i]) ;
			// printf("gl:%e\n", gl[i]);
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] = out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}

void cfftlog_modified_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, long N_pad){
	config my_config;
	my_config.nu = nu;
	my_config.c_window_width = c_window_width;
	my_config.derivative = derivative;
	my_config.N_pad = N_pad;
	cfftlog_modified_ells(x, fx, N, &my_config, ell, Nell, y, Fy);
}
// Only modification is g_l functions, when the integrand function is f(x)/(xy)^2
void cfftlog_modified_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		switch(config->derivative) {
			case 0: g_l_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 1: g_l_1_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 2: g_l_2_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
			default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
		}

		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i] ) ;
			// printf("gl:%e\n", gl[i]);
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] = out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}


void cfftlog_ells_increment(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		switch(config->derivative) {
			case 0: g_l(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 1: g_l_1(ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 2: g_l_2(ell[j], config->nu, eta_m, gl, halfN+1); break;
			default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
		}

		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i]) ;
			// printf("gl:%e\n", gl[i]);
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] += out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}
