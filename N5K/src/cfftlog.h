typedef struct config {
	double nu;
	double c_window_width;
	int derivative;
	long N_pad;
} config;

void cfftlog(double *x, double *fx, long N, config *config, double ell, double *y, double *Fy);

void cfftlog_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy);

void cfftlog_ells_increment(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy);

void cfftlog_wrapper(double *x, double *fx, long N, double ell, double *y, double *Fy, double nu, double c_window_width, int derivative, long N_pad);

void cfftlog_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, long N_pad);

void cfftlog_modified_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy);

void cfftlog_modified_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, long N_pad);
