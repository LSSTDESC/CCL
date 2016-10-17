#include "ccl_core.h"
#include "ccl_error.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

void ccl_check_status(ccl_cosmology *cosmo){
	switch (cosmo->status){
		case 0:
			return;
		// spacing allocation error, always terminate	
		case 2:		
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
		// spline allocation error, always terminate	
		case 4:		
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
		// implement softer error handling, e.g. for integral convergence here
		default:		
			fprintf(stderr,"%s",cosmo->status_message);
			exit(1);
	}
}
