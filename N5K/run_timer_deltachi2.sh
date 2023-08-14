#!/bin/bash

mkdir -p outputs

echo MATTER
echo 0.2
echo 0.2 >> outputs/times_MATTER_deltachi2.txt
python timer.py confs/deltachi2/config_matter_chi02.yml MATTER 10 none >> outputs/times_MATTER_deltachi2.txt
echo 0.7
echo 0.7 >> outputs/times_MATTER_deltachi2.txt
python timer.py confs/deltachi2/config_matter_chi07.yml MATTER 10 none >> outputs/times_MATTER_deltachi2.txt
echo 1.2
echo 1.2 >> outputs/times_MATTER_deltachi2.txt
python timer.py confs/deltachi2/config_matter_chi12.yml MATTER 10 none >> outputs/times_MATTER_deltachi2.txt
echo 1.7
echo 1.7 >> outputs/times_MATTER_deltachi2.txt
python timer.py confs/deltachi2/config_matter_chi17.yml MATTER 10 none >> outputs/times_MATTER_deltachi2.txt

#echo Levin
#echo 0.2
#echo 0.2 >> outputs/times_Levin_deltachi2.txt
#python timer.py confs/deltachi2/config_levin_fast_delta_chi2_0p2.yml Levin 10 none >> outputs/times_Levin_deltachi2.txt
#echo 0.7
#echo 0.7 >> outputs/times_Levin_deltachi2.txt
#python timer.py confs/deltachi2/config_levin_fast_delta_chi2_0p7.yml Levin 10 none >> outputs/times_Levin_deltachi2.txt
#echo 1.2
#echo 1.2 >> outputs/times_Levin_deltachi2.txt
#python timer.py confs/deltachi2/config_levin_fast_delta_chi2_1p2.yml Levin 10 none >> outputs/times_Levin_deltachi2.txt
#echo 1.7
#echo 1.7 >> outputs/times_Levin_deltachi2.txt
#python timer.py confs/deltachi2/config_levin_fast_delta_chi2_1p7.yml Levin 10 none >> outputs/times_Levin_deltachi2.txt
