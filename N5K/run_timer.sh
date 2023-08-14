#!/bin/bash

mkdir -p outputs

for nthr in 64 32 16 8 4 2 1
do
    export OMP_NUM_THREADS=${nthr}
    echo $nthr
    echo $OMP_NUM_THREADS >> outputs/times_FKEM.txt
    python timer.py confs/config_fkem.yml FKEM 10 none >> outputs/times_FKEM.txt
    echo $OMP_NUM_THREADS >> outputs/times_MATTER.txt
    python timer.py confs/config_matter.yml MATTER 10 none >> outputs/times_MATTER.txt
    echo $OMP_NUM_THREADS >> outputs/times_Levin.txt
    python timer.py confs/config_levin.yml Levin 10 none >> outputs/times_Levin.txt
done
unset OMP_NUM_THREADS
