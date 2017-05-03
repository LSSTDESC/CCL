#include "ccl_core.h"
#include "ccl_error.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

static CCLErrorHandlingPolicy _ccl_errorhandling_policy = CCL_ERRORHANDLING_POLICY_ABORT;

void
ccl_errorhandling_set_policy(CCLErrorHandlingPolicy policy)
{
    _ccl_errorhandling_policy = policy;
}

// RH 
void ccl_check_status(ccl_cosmology *cosmo, int * status){

    if (_ccl_errorhandling_policy == CCL_ERRORHANDLING_POLICY_UNWIND) {
        return;
    }

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

