#!/bin/bash
# N=15
# case="Case4"
# version="v3"
# D="../../../../OutlierGen/exos/${case}_${version}"
# bfname="10K_${case}"
# relpath="pickles/${case}_${version}"
# nsets=31
# wsize=1000


M=2
case="Case1"
D="../../../../OutlierGen/exos/small_cases/Case1"
bfname="1000_${case}"
relpath="pickles/small_cases/Case1"
nsets=2
wsize=100

for ((i=1; i<$nsets; i++))
do
	python3 cases.py --ex_number $i --nstreams $M --bfname $bfname --dfolder $D --relpath $relpath --wsize ${wsize}
done