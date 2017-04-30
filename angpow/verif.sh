#!/bin/bash
for i in 1 2 3 4 5 6
do
  ./bin/angpow angpow_bench${i}.ini >& tmp.log; rm tmp.log
  a=`paste angpow_bench${i}_cl.txt angpow_bench${i}_cl.txt.REF | gawk -f ./diff.awk`
  echo "Diff wrt bench${i} ref (l,max): $a"
done
