#include "ccl_core.h"
#include "ccl_error.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

// RH 
void ccl_check_status(ccl_cosmology *cosmo, int * status){
	switch (*status){
		case 0: //all good, nothing to do
			return;
		case CCL_ERROR_LINSPACE:	// spacing allocation error, always terminate		
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);	
		case CCL_ERROR_SPLINE:	// spline allocation error, always terminate	
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
		case CCL_ERROR_COMPUTECHI:	// compute_chi error //RH
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
                case CCL_ERROR_HMF_INTERP: // terminate if hmf definition not supported
                        fprintf(stderr,"%s",cosmo->status_message);
                        exit(1);

		// implement softer error handling, e.g. for integral convergence here
			
		default:		
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
	}
}
