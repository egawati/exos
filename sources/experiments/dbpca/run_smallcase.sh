#!/bin/bash
N=2
case="Case4"
D="../../../../OutlierGen/exos/small_cases/${case}"
bfname="100_${case}"
relpath="pickles/${case}"
nsets=2

for ((i=1; i<$nsets; i++))
do
	python3 nstreams.py --ex_number $i --nstreams $N --bfname $bfname --dfolder $D --relpath $relpath
done