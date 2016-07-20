//structure containing photo-z parameters
typedef struct pz_parameters {
	double sigma_z;
	//expand this as needed
};
//stucture containing tomopraphy parameters
typedef struct tomography_parameters {
	int N_tomo;
	double *z_lim;
	//expand this as needed
};
//function pointer to n(z) routine
typedef double (*n_of_z_function)(double z, int nz, pz_parameters *params, int *status);

typedef struct ccl_nz{
	pz_parameters pz;
	tomography_parameters tomo;
	n_of_z_function n_of_z;
}

// example n(z) to reproduce code comparison results
double n_of_z_gaussian(double z, int nz, pz_params *params,int *status){
  // bin width
  double sigma = 0.15;
  //bin center
  double x = 1.0+0.5*nz-z;
  return 1./sqrt(2.*M_PI*sigma*sigma)*exp(-x*x/(2.*sigma*sigma));
}