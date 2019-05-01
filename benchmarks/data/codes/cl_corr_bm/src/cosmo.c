#include "common.h"

static double speval_bis(double x,void *params)
{
  return spline_eval(x,(SplPar *)params);
}

typedef struct {
  Csm_params *cpar;
  double chi;
} Fpar;

static double fzero(double a,void *params)
{
  double chi=((Fpar *)params)->chi;
  double chia=csm_radial_comoving_distance(((Fpar *)params)->cpar,a);

  return chi-chia;
}

static double dfzero(double a,void *params)
{
  double h=csm_hubble(((Fpar *)params)->cpar,a);

  return 1./(a*a*h);
}

static void fdfzero(double a,void *params,double *f,double *df)
{
  *f=fzero(a,params);
  *df=dfzero(a,params);
}
  
static double a_of_chi(double chi,Csm_params *cpar,double *a_old,gsl_root_fdfsolver *s)
{
  if(chi==0)
    return 1.;
  else {
    Fpar p;
    gsl_function_fdf FDF;
    double a_previous,a_current=*a_old;
    
    p.cpar=cpar;
    p.chi=chi;
    FDF.f=&fzero;
    FDF.df=&dfzero;
    FDF.fdf=&fdfzero;
    FDF.params=&p;
    gsl_root_fdfsolver_set(s,&FDF,a_current);
    
    int iter=0,status;
    do {
      iter++;
      status=gsl_root_fdfsolver_iterate(s);
      a_previous=a_current;
      a_current=gsl_root_fdfsolver_root(s);
      status=gsl_root_test_delta(a_current,a_previous,1E-6,0);
    } while(status==GSL_CONTINUE);
    
    *a_old=a_current;
    return a_current;
  }
}

typedef struct {
  RunParams *par;
  double chi;
  int i_window;
} IntLensPar;

static double integrand_wm(double chip,void *params)
{
  IntLensPar *p=(IntLensPar *)params;
  double chi=p->chi;
  double z=spline_eval(chip,p->par->zofchi);
  double pz=spline_eval(z,p->par->wind_0[p->i_window]);
  double sz=spline_eval(z,p->par->sbias);
  double h=spline_eval(chip,p->par->hofchi);

  if(chi==0)
    return h*pz*0.5*(2-5*sz);
  else
    return h*pz*0.5*(2-5*sz)*(chip-chi)/chip;
}

static double window_magnification(double chi,RunParams *par,int i_window)
{
  double result,eresult;
  IntLensPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.par=par;
  ip.chi=chi;
  ip.i_window=i_window;
  F.function=&integrand_wm;
  F.params=&ip;
  gsl_integration_qag(&F,chi,par->chi_horizon,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);
  return result;
}

static double integrand_wl(double chip,void *params)
{
  IntLensPar *p=(IntLensPar *)params;
  double chi=p->chi;
  double z=spline_eval(chip,p->par->zofchi);
  double pz=spline_eval(z,p->par->wind_0[p->i_window]);
  double h=spline_eval(chip,p->par->hofchi);

  if(chi==0)
    return h*pz;
  else
    return h*pz*(chip-chi)/chip;
}

static double window_lensing(double chi,RunParams *par,int i_window)
{
  double result,eresult;
  IntLensPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.par=par;
  ip.chi=chi;
  ip.i_window=i_window;
  F.function=&integrand_wl;
  F.params=&ip;
  gsl_integration_qag(&F,chi,par->chi_horizon,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);
  return result;
}

#ifdef _DEBUG
#define NZ_BG 1024
static void print_bg(RunParams *par)
{
  int ii,icol;
  char fname[256];
  FILE *fo;
  double zmax=fmax(par->wind_0[0]->xf,par->wind_0[1]->xf);
  sprintf(fname,"%s_bg.db",par->prefix_out);
  fo=dam_fopen(fname,"w");
  fprintf(fo,"[0]z [1]chi [2]zb [3]a [4]h [5]gf [6]fg ");
  icol=6;
  if(par->do_nc) {
    fprintf(fo,"[%d]w0_1 [%d]w0_2 ",icol+1,icol+2);
    icol+=2;
    if(par->has_lensing) {
      fprintf(fo,"[%d]wM_1 [%d]wM_2 ",icol+1,icol+2);
      icol+=2;
    }
  }
  if(par->do_shear) {
    fprintf(fo,"[%d]wL_1 [%d]wL_2 ",icol+1,icol+2);
    icol+=2;
  }
  fprintf(fo,"\n");
  for(ii=0;ii<NZ_BG;ii++) {
    double z=zmax*(ii+0.5)/NZ_BG;
    double chi=csm_radial_comoving_distance(par->cpar,1./(1+z));
    fprintf(fo,"%lE %lE ",z,chi);
    fprintf(fo,"%lE ",spline_eval(chi,par->zofchi));
    fprintf(fo,"%lE ",spline_eval(chi,par->aofchi));
    fprintf(fo,"%lE ",spline_eval(chi,par->hofchi));
    fprintf(fo,"%lE ",spline_eval(chi,par->gfofchi));
    fprintf(fo,"%lE ",spline_eval(chi,par->fgofchi));
    if(par->do_nc) {
      fprintf(fo,"%lE ",spline_eval(z,par->wind_0[0]));
      fprintf(fo,"%lE ",spline_eval(z,par->wind_0[1]));
      if(par->has_lensing) {
	fprintf(fo,"%lE ",spline_eval(chi,par->wind_M[0]));
	fprintf(fo,"%lE ",spline_eval(chi,par->wind_M[1]));
      }
    }
    if(par->do_shear) {
      fprintf(fo,"%lE ",spline_eval(chi,par->wind_L[0]));
      fprintf(fo,"%lE ",spline_eval(chi,par->wind_L[1]));
    }
    fprintf(fo,"\n");
  }
  fclose(fo);
}
#endif //_DEBUG

RunParams *init_params(char *fname_ini)
{
  FILE *fi;
  int n,ii,stat,ibin;
  double *x,*a,*y,dchi;
  RunParams *par=param_new();
  par->cpar=csm_params_new();
  read_parameter_file(fname_ini,par);

  csm_unset_gsl_eh(par->cpar);
  if(par->has_bg) {
    double hub;
    csm_background_set(par->cpar,par->om,par->ol,par->ob,par->w0,par->wa,par->h0,D_TCMB);
    par->chi_horizon=csm_radial_comoving_distance(par->cpar,0.);
    par->chi_LSS=csm_radial_comoving_distance(par->cpar,1./(1+D_Z_REC));
    hub=csm_hubble(par->cpar,1.);
    par->prefac_lensing=1.5*hub*hub*par->om;
    
    n=(int)(par->chi_horizon/par->dchi)+1;
    dchi=par->chi_horizon/n;
    par->dchi=dchi;
    
    x=(double *)dam_malloc(n*sizeof(double));
    a=(double *)dam_malloc(n*sizeof(double));
    y=(double *)dam_malloc(n*sizeof(double));
    
    for(ii=0;ii<n;ii++)
      x[ii]=dchi*ii;
    
    printf("Setting up background splines\n");
    //Set chi <-> a correspondence
    const gsl_root_fdfsolver_type *T=gsl_root_fdfsolver_newton;
    gsl_root_fdfsolver *s=gsl_root_fdfsolver_alloc(T);
    double a_old=1.0;
    for(ii=0;ii<n;ii++)
      a[ii]=a_of_chi(x[ii],par->cpar,&a_old,s);
    gsl_root_fdfsolver_free(s);
    par->aofchi=spline_init(n,x,a,1.0,0.0);
    
    //Compute redshift
    for(ii=0;ii<n;ii++)
      y[ii]=1./a[ii]-1;
    par->zofchi=spline_init(n,x,y,y[0],y[n-1]);
    
    //Compute hubble scale
    for(ii=0;ii<n;ii++)
      y[ii]=csm_hubble(par->cpar,a[ii]);
    par->hofchi=spline_init(n,x,y,y[0],y[n-1]);
    
    //Compute growth factor
    double g0=csm_growth_factor(par->cpar,1.0);
    for(ii=0;ii<n;ii++)
      y[ii]=csm_growth_factor(par->cpar,a[ii])/g0;
    par->gfofchi=spline_init(n,x,y,1.,0.);
    
    //Compute growth rate
    for(ii=0;ii<n;ii++)
      y[ii]=csm_f_growth(par->cpar,a[ii]);
    par->fgofchi=spline_init(n,x,y,y[0],1.);
    free(x); free(a); free(y);
  }

  //Allocate power spectra
  if(par->do_nc) {
    par->cl_dd=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    if(par->do_shear) {
      par->cl_d1l2=(double *)dam_malloc((par->lmax+1)*sizeof(double));
      par->cl_d2l1=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    }
    if(par->do_cmblens)
      par->cl_dc=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    if(par->do_isw)
      par->cl_di=(double *)dam_malloc((par->lmax+1)*sizeof(double));
  }
  if(par->do_shear) {
    par->cl_ll=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    if(par->do_cmblens)
      par->cl_lc=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    if(par->do_isw)
      par->cl_li=(double *)dam_malloc((par->lmax+1)*sizeof(double));
  }
  if(par->do_cmblens) {
    par->cl_cc=(double *)dam_malloc((par->lmax+1)*sizeof(double));
    if(par->do_isw)
      par->cl_ci=(double *)dam_malloc((par->lmax+1)*sizeof(double));
  }
  if(par->do_isw)
    par->cl_ii=(double *)dam_malloc((par->lmax+1)*sizeof(double));

  if(par->do_w_theta) {
    if(par->do_nc) {
      par->wt_dd=(double *)dam_malloc(par->n_th*sizeof(double));
      if(par->do_shear) {
	par->wt_d1l2=(double *)dam_malloc(par->n_th*sizeof(double));
	par->wt_d2l1=(double *)dam_malloc(par->n_th*sizeof(double));
      }
      if(par->do_cmblens)
	par->wt_dc=(double *)dam_malloc(par->n_th*sizeof(double));
      if(par->do_isw)
	par->wt_di=(double *)dam_malloc(par->n_th*sizeof(double));
    }
    if(par->do_shear) {
      par->wt_ll_pp=(double *)dam_malloc(par->n_th*sizeof(double));
      par->wt_ll_mm=(double *)dam_malloc(par->n_th*sizeof(double));
      if(par->do_cmblens)
	par->wt_lc=(double *)dam_malloc(par->n_th*sizeof(double));
      if(par->do_isw)
	par->wt_li=(double *)dam_malloc(par->n_th*sizeof(double));
    }
    if(par->do_cmblens) {
      par->wt_cc=(double *)dam_malloc(par->n_th*sizeof(double));
      if(par->do_isw)
	par->wt_ci=(double *)dam_malloc(par->n_th*sizeof(double));
    }
    if(par->do_isw)
      par->wt_ii=(double *)dam_malloc(par->n_th*sizeof(double));
  }
    
  if(par->do_nc || par->do_shear || par->do_cmblens || par->do_isw)
    csm_set_linear_pk(par->cpar,par->fname_pk,D_LKMIN,D_LKMAX,0.01,par->ns,par->s8);
  
  if(par->do_nc || par->do_shear) {
    par->wind_0=dam_malloc(2*sizeof(SplPar *));
    for(ibin=0;ibin<2;ibin++) {
      printf("Reading window function %s\n",par->fname_window[ibin]);
      fi=dam_fopen(par->fname_window[ibin],"r");
      n=dam_linecount(fi); rewind(fi);
      //Read unnormalized window
      x=(double *)dam_malloc(n*sizeof(double));
      y=(double *)dam_malloc(n*sizeof(double));
      for(ii=0;ii<n;ii++) {
	stat=fscanf(fi,"%lE %lE",&(x[ii]),&(y[ii]));
	if(stat!=2)
	  dam_report_error(1,"Error reading file, line %d\n",ii+1);
      }
      fclose(fi);
      par->wind_0[ibin]=spline_init(n,x,y,0.,0.);
      //Normalize window
      double norm,enorm;
      gsl_function F;
      gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
      F.function=&speval_bis;
      F.params=par->wind_0[ibin];
      gsl_integration_qag(&F,x[0],x[n-1],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&norm,&enorm);
      gsl_integration_workspace_free(w);
      for(ii=0;ii<n;ii++)
      	y[ii]/=norm;
      spline_free(par->wind_0[ibin]);
      par->wind_0[ibin]=spline_init(n,x,y,0.,0.);
      free(x); free(y);
    }
  }

  if(par->do_nc) {
    if(par->has_dens==1) {
      printf("Reading bias function %s\n",par->fname_bias);
      fi=dam_fopen(par->fname_bias,"r");
      n=dam_linecount(fi); rewind(fi);
      //Read bias
      x=(double *)dam_malloc(n*sizeof(double));
      y=(double *)dam_malloc(n*sizeof(double));
      for(ii=0;ii<n;ii++) {
	stat=fscanf(fi,"%lE %lE",&(x[ii]),&(y[ii]));
	if(stat!=2)
	  dam_report_error(1,"Error reading file, line %d\n",ii+1);
      }
      fclose(fi);
      par->bias=spline_init(n,x,y,y[0],y[n-1]);
      free(x); free(y);
    }

    if(par->has_lensing==1) {
      printf("Reading s-bias function %s\n",par->fname_sbias);
      fi=dam_fopen(par->fname_sbias,"r");
      n=dam_linecount(fi); rewind(fi);
      //Read s-bias
      x=(double *)dam_malloc(n*sizeof(double));
      y=(double *)dam_malloc(n*sizeof(double));
      for(ii=0;ii<n;ii++) {
	stat=fscanf(fi,"%lE %lE",&(x[ii]),&(y[ii]));
	if(stat!=2)
	  dam_report_error(1,"Error reading file, line %d\n",ii+1);
      }
      fclose(fi);
      par->sbias=spline_init(n,x,y,y[0],y[n-1]);
      free(x); free(y);
    
      printf("Computing lensing magnification window function\n");
      par->wind_M=dam_malloc(2*sizeof(SplPar *));
      for(ibin=0;ibin<2;ibin++) {
	double dchi_here;
	double zmax=par->wind_0[ibin]->xf;
	double chimax=csm_radial_comoving_distance(par->cpar,1./(1+zmax));
	n=(int)(chimax/par->dchi)+1;
	dchi_here=chimax/n;
      
	x=(double *)dam_malloc(n*sizeof(double));
	y=(double *)dam_malloc(n*sizeof(double));

#ifdef _HAVE_OMP
#pragma omp parallel default(none) shared(n,x,y,par,dchi_here,ibin) 
	{
#endif //_HAVE_OMP
	  int j;
	  
#ifdef _HAVE_OMP
#pragma omp for	
#endif //_HAVE_OMP
	  for(j=0;j<n;j++) {
	    x[j]=dchi_here*j;
	    y[j]=window_magnification(x[j],par,ibin);
	  } //end omp for
#ifdef _HAVE_OMP
	} //end omp parallel
#endif //_HAVE_OMP
	par->wind_M[ibin]=spline_init(n,x,y,y[0],0);
	free(x); free(y);
      }
    }
  }

  if(par->do_shear) {
    printf("Computing lensing window function\n");
    par->wind_L=dam_malloc(2*sizeof(SplPar *));
    for(ibin=0;ibin<2;ibin++) {
      double dchi_here;
      double zmax=par->wind_0[ibin]->xf;
      double chimax=csm_radial_comoving_distance(par->cpar,1./(1+zmax));
      n=(int)(chimax/par->dchi)+1;
      dchi_here=chimax/n;
    
      x=(double *)dam_malloc(n*sizeof(double));
      y=(double *)dam_malloc(n*sizeof(double));
    
#ifdef _HAVE_OMP
#pragma omp parallel default(none) shared(n,x,y,par,dchi_here,ibin) 
      {
#endif //_HAVE_OMP
	int j;
	
#ifdef _HAVE_OMP
#pragma omp for	
#endif //_HAVE_OMP
	for(j=0;j<n;j++) {
	  x[j]=dchi_here*j;
	  y[j]=window_lensing(x[j],par,ibin);
	}
#ifdef _HAVE_OMP
      } //end omp parallel
#endif //_HAVE_OMP
      par->wind_L[ibin]=spline_init(n,x,y,y[0],0);
      free(x); free(y);
    }
  }

#ifdef _DEBUG
  print_bg(par);
#endif //_DEBUG
  return par;
}
