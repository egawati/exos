#!/bin/bash
N=15
case="Case1"
D="../../../../OutlierGen/exos/${case}"
bfname="10K_${case}"
relpath="pickles/${case}"
nsets=31

for ((i=1; i<$nsets; i++))
do
	python3 cases.py --ex_number $i --nstreams $N --bfname $bfname --dfolder $D --relpath $relpath
done