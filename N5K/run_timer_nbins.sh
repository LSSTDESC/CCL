#!/bin/bash

mkdir -p outputs

for i in 10 9 8 7 6 5 4 3 2 1
do
    echo ${i}
    echo FKEM
    python timer_nbins.py confs/config_fkem.yml FKEM ${i} 5 >> outputs/times_nb_sh5_FKEM.txt
    python timer_nbins.py confs/config_fkem.yml FKEM ${i} 1 >> outputs/times_nb_sh1_FKEM.txt
    exit 1
    
    echo MATTER
    python timer_nbins.py confs/config_matter.yml MATTER ${i} 5 >> outputs/times_nb_sh5_MATTER.txt
    python timer_nbins.py confs/config_matter.yml MATTER ${i} 1 >> outputs/times_nb_sh1_MATTER.txt
    echo Levin
    python timer_nbins.py confs/config_levin.yml Levin ${i} 5 >> outputs/times_nb_sh5_Levin.txt
    python timer_nbins.py confs/config_levin.yml Levin ${i} 1 >> outputs/times_nb_sh1_Levin.txt
done

for i in 5 4 3 2 1
do
    echo ${i}
    echo FKEM
    python timer_nbins.py confs/config_fkem.yml FKEM 10 ${i} >> outputs/times_nb_cl10_FKEM.txt
    python timer_nbins.py confs/config_fkem.yml FKEM 1 ${i} >> outputs/times_nb_cl1_FKEM.txt
    echo MATTER
    python timer_nbins.py confs/config_matter.yml MATTER 10 ${i} >> outputs/times_nb_cl10_MATTER.txt
    python timer_nbins.py confs/config_matter.yml MATTER 1 ${i} >> outputs/times_nb_cl1_MATTER.txt
    echo Levin
    python timer_nbins.py confs/config_levin.yml Levin 10 ${i} >> outputs/times_nb_cl10_Levin.txt
    python timer_nbins.py confs/config_levin.yml Levin 1 ${i} >> outputs/times_nb_cl1_Levin.txt
done
