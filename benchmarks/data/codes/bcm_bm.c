/** 
    This file contains the modification made to the hi_class code
    (1st anniversary release - https://github.com/miguelzuma/hi_class_public/releases/tag/v1.1)
    to directly output the baryon-corrected matter power spectrum.
    Note that this is an excerpt from the actual code, and therefore won't
    run on its own.
**/
...


    //This would be appended after line 169 of nonlinear.c in the version above.
    //add the baryonic effects on top
    if (pnl->method == nl_baryon) {
      if (pnl->nonlinear_verbose > 0)
	printf("Adding baryonic effects (eq 4.2 of 1510.06034) \n");

      /** - loop over time to add baryonic correction */

      for (index_tau = pnl->tau_size-1; index_tau>=0; index_tau--) {

	/* get P_L(k) at this time */
	class_call(nonlinear_pk_l(ppt,ppm,pnl,index_tau,pk_l),
		  pnl->error_message,
		  pnl->error_message);
	
	class_alloc(pvecback,pba->bg_size*sizeof(double),pnl->error_message);

	class_call(background_at_tau(pba,pnl->tau[index_tau],pba->long_info,pba->inter_normal,&last_index,pvecback),
		  pba->error_message,
		  pnl->error_message);

	/* correct the spectrum */
	for (index_k=0; index_k<pnl->k_size; index_k++) {
	  
	  /*Section 4.6 of 1510.06034
	  * NOTE: factors of h!
	  */
	  B = 0.105*log10(pnl->baryon_M_c) - 1.27;
	  B /= (1.+pow((1./pvecback[pba->index_bg_a]-1.)/2.3, 2.5));
	  
	  k_g = pba->h*0.7*pow(1.-B, 4)*pow(pnl->baryon_eta_b,-1.6);
	  
	  G = 1. + B*(1./(1. + pow(pnl->k[index_k]/k_g,3)) -1.);
	  
	  //multiply by the stellar correction
	  G *= 1. + pow(pnl->k[index_k]/55./pba->h,2);
	  
	  // sqrt(pk_nl[index_k]/pk_l[index_k])
	  pnl->nl_corr_density[index_tau * pnl->k_size + index_k] *= sqrt(G);
	}
      }
      
    }

...
