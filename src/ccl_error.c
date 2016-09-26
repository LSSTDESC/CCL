void ccl_check_status(ccl_cosmology *cosmo){
	if (cosmo->status){
		fprintf(stderr,cosmo->status_message);
		EXIT(1);
	}
}