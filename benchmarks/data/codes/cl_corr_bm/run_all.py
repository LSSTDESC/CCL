import os
#Produces all angular power spectrum benchmarks
#Contact david.alonso@physics.ox.ac.uk if you have issues running this script

def run_task(window_type,bins_type,b1,b2) :
    do_logbin=0
    if bins_type=="lin" :
        do_logbin=0
    elif bins_type=="log" :
        do_logbin=1
    else :
        print("Wrong binning scheme "+bins_type)

    strout="#Cosmological parameters\n"
    strout+="omega_m= 0.3\n"
    strout+="omega_l= 0.7\n"
    strout+="omega_b= 0.0\n"
    strout+="w0= -1.\n"
    strout+="wa= 0.\n"
    strout+="h= 0.7\n"
    strout+="ns= 0.96\n"
    strout+="s8= 0.8\n"
    strout+="\n"
    strout+="#Radial resolution\n"
    strout+="d_chi= 5.\n"
    strout+="\n"
    strout+="#Maximum multipole\n"
    strout+="l_max= 30000\n"
    strout+="\n"
    strout+="#Behavioural flags (include number counts? include lensing shear? include CMB lensing?)\n"
    strout+="do_nc= 1\n"
    strout+="has_nc_dens= 1\n"
    strout+="has_nc_rsd= 0\n"
    strout+="has_nc_lensing= 0\n"
    strout+="do_shear= 1\n"
    strout+="do_cmblens= 1\n"
    strout+="\n"
    strout+="#Angular correlation function\n"
    strout+="do_w_theta= 1\n"
    strout+="use_logbin= %d\n"%do_logbin
    strout+="theta_min= 0\n"
    strout+="theta_max= 10.\n"
    strout+="n_bins_theta= 15\n"
    strout+="n_bins_decade= 5\n"
    strout+="\n"
    strout+="#File names (window function, bias, magnification bias, power spectrum)\n"
    strout+="window_1_fname= curves/bin%d_"%b1+window_type+".txt\n"
    strout+="window_2_fname= curves/bin%d_"%b2+window_type+".txt\n"
    strout+="bias_fname= curves/bias.txt\n"
    strout+="sbias_fname= nothing\n"
    strout+="pk_fname= BBKS\n"
    strout+="\n"
    strout+="#Output prefix\n"
    strout+="prefix_out= output/run_b%d"%b1+"b%d"%b2+window_type+"_"+bins_type

    parname="param_limberjack_b%d"%b1+"b%d"%b2+window_type+"_"+bins_type+".ini"
    f=open(parname,"w")
    f.write(strout)
    f.close()

    os.system("./limberjack "+parname)

run_task("analytic","log",1,1)
run_task("analytic","log",1,2)
run_task("analytic","log",2,2)
run_task("histo","log",1,1)
run_task("histo","log",1,2)
run_task("histo","log",2,2)
