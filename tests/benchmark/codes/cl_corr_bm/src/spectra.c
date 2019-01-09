#include "common.h"

typedef struct {
  int l;
  RunParams *par;
  char *tr1;
  char *tr2;
} IntClPar;

static double cl_integrand(double lk,void *params)
{
  double d1,d2;
  IntClPar *p=(IntClPar *)params;
  double k=pow(10.,lk);
  double pk=csm_Pk_linear_0(p->par->cpar,k);
  d1=transfer_wrap(p->l,k,p->par,p->tr1,0);
  d2=transfer_wrap(p->l,k,p->par,p->tr2,1);

  return k*d1*d2*pk;
}

static double spectra(char *tr1,char *tr2,int l,RunParams *par)
{
  IntClPar ipar;
  double result=0,eresult;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
  ipar.l=l;
  ipar.par=par;
  ipar.tr1=tr1;
  ipar.tr2=tr2;
  F.function=&cl_integrand;
  F.params=&ipar;
  gsl_integration_qag(&F,D_LKMIN,D_LKMAX,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);

  return M_LN10*2*result/(2*l+1.);
}

void compute_spectra(RunParams *par)
{
  printf("Computing power spectra\n");
#ifdef _HAS_OMP
#pragma omp parallel default(none) shared(par)
  {
#endif //_HAS_OMP
    int l;
#ifdef _HAS_OMP
#pragma omp for
#endif //_HAS_OMP
    for(l=0;l<=par->lmax;l++) {
#ifdef _DEBUG
      printf("%d \n",l);
#endif //_DEBUG
      if(par->do_nc) {
	par->cl_dd[l]=spectra("nc","nc",l,par);
	if(par->do_shear) {
	  par->cl_d1l2[l]=spectra("nc","shear",l,par);
	  par->cl_d2l1[l]=spectra("shear","nc",l,par);
	}
	if(par->do_cmblens)
	  par->cl_dc[l]=spectra("nc","cmblens",l,par);
	if(par->do_isw)
	  par->cl_di[l]=spectra("nc","isw",l,par);
      }
      if(par->do_shear) {
	par->cl_ll[l]=spectra("shear","shear",l,par);
	if(par->do_cmblens)
	  par->cl_lc[l]=spectra("shear","cmblens",l,par);
	if(par->do_isw)
	  par->cl_li[l]=spectra("shear","isw",l,par);
      }
      if(par->do_cmblens) {
	par->cl_cc[l]=spectra("cmblens","cmblens",l,par);
	if(par->do_isw)
	  par->cl_ci[l]=spectra("cmblens","isw",l,par);
      }
      if(par->do_isw)
	par->cl_ii[l]=spectra("isw","isw",l,par);
    } //end omp for
#ifdef _HAS_OMP
  } //end omp parallel
#endif //_HAS_OMP
}

typedef struct {
  int i_bessel;
  RunParams *par;
  double th;
  SplPar *clsp;
  double *cl;
} IntWtPar;

static double wt_integrand(double l,void *params)
{
  IntWtPar *p=(IntWtPar *)params;
  double x=l*p->th;
  //  double cl=spline_eval(l,p->clsp);
  double cl=p->cl[(int)l];
  double jbes;

  if(p->i_bessel)
    jbes=gsl_sf_bessel_Jn(p->i_bessel,x);
  else
    jbes=gsl_sf_bessel_J0(x);

  return l*jbes*cl;
}

static void compute_wt_single(RunParams *par,double *cl,double *wt,double *llist,int bessel_order)
{
#ifdef _HAS_OMP
#pragma omp parallel default(none)		\
  shared(par,cl,wt,llist,bessel_order)
  {
#endif //_HAS_OMP
    int ith;
    double result,eresult;
    gsl_function F;
    gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
    SplPar *clsp=spline_init((par->lmax+1),llist,cl,0,0);
    IntWtPar ipar;

    ipar.i_bessel=bessel_order;
    ipar.par=par;
    ipar.clsp=clsp;
    ipar.cl=cl;
    
#ifdef _HAS_OMP
#pragma omp for
#endif //_HAS_OMP
    for(ith=0;ith<par->n_th;ith++) {
      if(par->do_w_theta_logbin)
	ipar.th=DTOR*par->th_max*pow(10.,(ith+0.5-par->n_th)/par->n_th_logint);
      else
	ipar.th=DTOR*(par->th_min+(par->th_max-par->th_min)*(ith+0.5)/par->n_th);
      F.function=&wt_integrand;
      F.params=&ipar;
      gsl_integration_qag(&F,llist[0],llist[par->lmax],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
      wt[ith]=result/(2*M_PI);
    }//end omp for
    gsl_integration_workspace_free(w);
    spline_free(clsp);
#ifdef _HAS_OMP
  } //end omp parallel
#endif //_HAS_OMP
}

void compute_w_theta(RunParams *par)
{
  if(par->do_w_theta) {
    int l;
    double *llist;

    printf("Computing correlation functions\n");

    llist=dam_malloc((par->lmax+1)*sizeof(double));
    for(l=0;l<=par->lmax;l++)
      llist[l]=(float)l;

    if(par->do_nc) {
      compute_wt_single(par,par->cl_dd,par->wt_dd,llist,0);
      if(par->do_shear) {
	compute_wt_single(par,par->cl_d1l2,par->wt_d1l2,llist,0);
	compute_wt_single(par,par->cl_d2l1,par->wt_d2l1,llist,0);
      }
      if(par->do_cmblens)
	compute_wt_single(par,par->cl_dc,par->wt_dc,llist,0);
      if(par->do_isw)
	compute_wt_single(par,par->cl_di,par->wt_di,llist,0);
    }
    if(par->do_shear) {
      compute_wt_single(par,par->cl_ll,par->wt_ll_pp,llist,0);
      compute_wt_single(par,par->cl_ll,par->wt_ll_mm,llist,4);
      if(par->do_cmblens)
	compute_wt_single(par,par->cl_lc,par->wt_lc,llist,0);
      if(par->do_isw)
	compute_wt_single(par,par->cl_li,par->wt_li,llist,0);
    }
    if(par->do_cmblens) {
      compute_wt_single(par,par->cl_cc,par->wt_cc,llist,0);
      if(par->do_isw)
	compute_wt_single(par,par->cl_ci,par->wt_ci,llist,0);
    }
    if(par->do_isw)
      compute_wt_single(par,par->cl_ii,par->wt_ii,llist,0);
    free(llist);
  }
  else {
    printf("Skipping correlation functions\n");
  }
}
