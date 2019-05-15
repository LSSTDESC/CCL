#include "common.h"

static void write_wt_single(int n_th,int n_th_logint,int th_min,int th_max,int do_logbin,
			    double *wt,char *prefix,char *suffix)
{
  int ith;
  FILE *fo;
  char fname[256];

  sprintf(fname,"%s_wt_%s.txt",prefix,suffix);
  fo=dam_fopen(fname,"w");
  for(ith=0;ith<n_th;ith++) {
    double th;
    if(do_logbin)
      th=th_max*pow(10.,(ith+0.5-n_th)/n_th_logint);
    else
      th=th_min+(th_max-th_min)*(ith+0.5)/n_th;
    fprintf(fo,"%lE %lE\n",th,wt[ith]);
  }
  fclose(fo);
}

static void write_cl_single(int lmax,double *cl,char *prefix,char *suffix)
{
  int l;
  FILE *fo;
  char fname[256];

  sprintf(fname,"%s_cl_%s.txt",prefix,suffix);
  fo=dam_fopen(fname,"w");
  for(l=0;l<=lmax;l++)
    fprintf(fo,"%d %lE\n",l,cl[l]);
  fclose(fo);
}

void write_output(RunParams *par)
{
  if(par->do_nc) {
    write_cl_single(par->lmax,par->cl_dd,par->prefix_out,"dd");
    if(par->do_shear) {
      write_cl_single(par->lmax,par->cl_d1l2,par->prefix_out,"d1l2");
      write_cl_single(par->lmax,par->cl_d2l1,par->prefix_out,"d2l1");
    }
    if(par->do_cmblens)
      write_cl_single(par->lmax,par->cl_dc,par->prefix_out,"dc");
    if(par->do_isw)
      write_cl_single(par->lmax,par->cl_di,par->prefix_out,"di");
  }
  if(par->do_shear) {
    write_cl_single(par->lmax,par->cl_ll,par->prefix_out,"ll");
    if(par->do_cmblens)
      write_cl_single(par->lmax,par->cl_lc,par->prefix_out,"lc");
    if(par->do_isw)
      write_cl_single(par->lmax,par->cl_li,par->prefix_out,"li");
  }
  if(par->do_cmblens) {
    write_cl_single(par->lmax,par->cl_cc,par->prefix_out,"cc");
    if(par->do_isw)
      write_cl_single(par->lmax,par->cl_ci,par->prefix_out,"ci");
  }
  if(par->do_isw)
    write_cl_single(par->lmax,par->cl_ii,par->prefix_out,"ii");

  if(par->do_w_theta) {
    if(par->do_nc) {
      write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
		      par->do_w_theta_logbin,par->wt_dd,par->prefix_out,"dd");
      if(par->do_shear) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_d1l2,par->prefix_out,"d1l2");
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_d2l1,par->prefix_out,"d2l1");
      }
      if(par->do_cmblens) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_dc,par->prefix_out,"dc");
      }
      if(par->do_isw) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_di,par->prefix_out,"di");
      }
    }
    if(par->do_shear) {
      write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
		      par->do_w_theta_logbin,par->wt_ll_pp,par->prefix_out,"ll_pp");
      write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
		      par->do_w_theta_logbin,par->wt_ll_mm,par->prefix_out,"ll_mm");
      if(par->do_cmblens) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_lc,par->prefix_out,"lc");
      }
      if(par->do_isw) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_li,par->prefix_out,"li");
      }
    }
    if(par->do_cmblens) {
      write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
		      par->do_w_theta_logbin,par->wt_cc,par->prefix_out,"cc");
      if(par->do_isw) {
	write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
			par->do_w_theta_logbin,par->wt_ci,par->prefix_out,"ci");
      }
    }
    if(par->do_isw) {
      write_wt_single(par->n_th,par->n_th_logint,par->th_min,par->th_max,
		      par->do_w_theta_logbin,par->wt_ii,par->prefix_out,"ii");
    }
  }
}
 
int read_parameter_file(char *fname,RunParams *par)
{
  FILE *fi;
  int n_lin,ii;

  //Read parameters from file
  fi=dam_fopen(fname,"r");
  n_lin=dam_linecount(fi); rewind(fi);
  for(ii=0;ii<n_lin;ii++) {
    char s0[512],s1[64],s2[256];
    if(fgets(s0,sizeof(s0),fi)==NULL)
      dam_report_error(1,"Error reading line %d, file %s\n",ii+1,fname);
    if((s0[0]=='#')||(s0[0]=='\n')||(s0[0]==' ')) continue;
    int sr=sscanf(s0,"%s %s",s1,s2);
    if(sr!=2)
      dam_report_error(1,"Error reading line %d, file %s\n",ii+1,fname);

    if(!strcmp(s1,"omega_m="))
      par->om=atof(s2);
    else if(!strcmp(s1,"omega_l="))
      par->ol=atof(s2);
    else if(!strcmp(s1,"omega_b="))
      par->ob=atof(s2);
    else if(!strcmp(s1,"w0="))
      par->w0=atof(s2);
    else if(!strcmp(s1,"wa="))
      par->wa=atof(s2);
    else if(!strcmp(s1,"h="))
      par->h0=atof(s2);
    else if(!strcmp(s1,"ns="))
      par->ns=atof(s2);
    else if(!strcmp(s1,"s8="))
      par->s8=atof(s2);
    else if(!strcmp(s1,"d_chi="))
      par->dchi=atof(s2);
    else if(!strcmp(s1,"l_max="))
      par->lmax=atoi(s2);
    else if(!strcmp(s1,"do_nc="))
      par->do_nc=atoi(s2);
    else if(!strcmp(s1,"has_nc_dens="))
      par->has_dens=atoi(s2);
    else if(!strcmp(s1,"has_nc_rsd="))
      par->has_rsd=atoi(s2);
    else if(!strcmp(s1,"has_nc_lensing="))
      par->has_lensing=atoi(s2);
    else if(!strcmp(s1,"do_shear="))
      par->do_shear=atoi(s2);
    else if(!strcmp(s1,"do_cmblens="))
      par->do_cmblens=atoi(s2);
    else if(!strcmp(s1,"do_isw="))
      par->do_isw=atoi(s2);
    else if(!strcmp(s1,"do_w_theta="))
      par->do_w_theta=atoi(s2);
    else if(!strcmp(s1,"use_logbin="))
      par->do_w_theta_logbin=atoi(s2);
    else if(!strcmp(s1,"theta_min="))
      par->th_min=atof(s2);
    else if(!strcmp(s1,"theta_max="))
      par->th_max=atof(s2);
    else if(!strcmp(s1,"n_bins_theta="))
      par->n_th=atof(s2);
    else if(!strcmp(s1,"n_bins_decade="))
      par->n_th_logint=atof(s2);
    else if(!strcmp(s1,"window_1_fname="))
      sprintf(par->fname_window[0],"%s",s2);
    else if(!strcmp(s1,"window_2_fname="))
      sprintf(par->fname_window[1],"%s",s2);
    else if(!strcmp(s1,"bias_fname="))
      sprintf(par->fname_bias,"%s",s2);
    else if(!strcmp(s1,"sbias_fname="))
      sprintf(par->fname_sbias,"%s",s2);
    else if(!strcmp(s1,"pk_fname="))
      sprintf(par->fname_pk,"%s",s2);
    else if(!strcmp(s1,"prefix_out="))
      sprintf(par->prefix_out,"%s",s2);
    else
      dam_report_error(0,"Unknown parameter %s\n",s1);
  }
  fclose(fi);

  if(par->do_w_theta==0)
    par->do_w_theta_logbin=0;
  
  par->has_bg=1;

  return 0;
}
