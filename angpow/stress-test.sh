#!/bin/bash

benches=(1 2 3 4)

nthreads=(1 2 4 8 16)
ntries=10

#echo "OMP    & 1   &  2   & 4  &  8  &  16 \\\\"
echo ${nthreads[*]} | awk 'BEGIN{printf "OMP\t&"} {for (i=1;i<NF;i++) printf $i"\t&"; printf $NF"\t""\\\\\n"}'
for i in ${benches[*]}
do
  for n in ${nthreads[*]}
  do
    export OMP_NUM_THREADS=$n
    sum=0
    for c in `seq 1 $ntries`
    do
      ./bin/angpow angpow_bench${i}.ini > tmp.log
      a=$(grep Total tmp.log | awk '{print $NF}')
      sum=`echo ${a/s/} | gawk -v s=$sum '{print $1+s}'`
    done
    avg=`echo $sum | gawk -v ntry=$ntries '{printf("%5.2f",$1/ntry)}'`
    t[n]=$avg
  done
echo ${t[*]} | awk -v i=$i 'BEGIN{printf "Test " i"\t&"} {for (i=1;i<NF;i++) printf $i"\t&"; printf $NF"\t""\\\\\n"}'
done
