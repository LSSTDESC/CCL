#include "ccl_core.h"
#include "ccl_error.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"


void clear_exception(){
    // Clear the global error state
	global_error_state = 0;
}

void throw_exception(int err, char* msg){
    // Set global error state to the errorcode of this exception
    global_error_state = err;
    
    // Copy error message
    strncpy(global_error_message, msg, 512);
    
    // Exit if fatal errors are enabled
    if ((!global_error_continue) && (err)){
        fprintf(stderr, "%s\n", global_error_message);
        exit(err);
    }
}

int check_exception(){
    // Check the global error state for an error code
	return global_error_state;
}

char* get_error_message(void){
    // Return the global error message
    return &global_error_message;
}

void set_continue_on_error(){
    // Set global error continue flag
    global_error_continue = 1;
}


void ccl_check_status(ccl_cosmology *cosmo, int * status){
	switch (*status){
		case 0: //all good, nothing to do
			return;
		/*
		case CCL_ERROR_LINSPACE:// spacing allocation error, always terminate		
			throw_exception(CCL_ERROR_LINSPACE, cosmo->status_message);
		case CCL_ERROR_SPLINE:	// spline allocation error, always terminate	
			throw_exception(CCL_ERROR_SPLINE, cosmo->status_message);
		case CCL_ERROR_COMPUTECHI:	// compute_chi error
			throw_exception(CCL_ERROR_COMPUTECHI, cosmo->status_message);
        case CCL_ERROR_HMF_INTERP: // terminate if hmf definition not supported
            throw_exception(CCL_ERROR_HMF_INTERP, cosmo->status_message);
        */
		// implement softer error handling, e.g. for integral convergence here
			
		default:
			throw_exception(*status, cosmo->status_message);
	}
}
