#!/bin/bash
N=15
case="Case1"
version="v3"
D="../../../../OutlierGen/exos/${case}_${version}"
bfname="10K_${case}"
windows=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)
nsets=16
ex_num=1
relpath="pickles/${case}_${version}/windows/exp_${ex_num}"

for ((i=1; i<$nsets; i++))
do
	python3 cases.py --ex_number $ex_num --nstreams $N --bfname $bfname --dfolder $D --relpath $relpath --wsize ${windows[$i-1]}
done