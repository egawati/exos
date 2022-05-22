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
D="../../../../OutlierGen/exos/small_cases/${case}"
bfname="w1K_${case}"
relpath="pickles/small_cases/${case}"
nsets=1
wsize=1000
mu=5.0
sigma=0.5

for ((i=1; i<$nsets+1; i++))
do
	python3 cases.py --ex_number $i --nstreams $M --bfname $bfname --dfolder $D --relpath $relpath --wsize ${wsize} --init_mu ${mu} --init_sigma ${sigma}
done

# M=2
# case="Case4"
# D="../../../../OutlierGen/exos/small_cases/${case}"
# bfname="1K_${case}"
# relpath="pickles/small_cases/${case}"
# nsets=1
# wsize=100

# for ((i=1; i<$nsets+1; i++))
# do
# 	python3 cases.py --ex_number $i --nstreams $M --bfname $bfname --dfolder $D --relpath $relpath --wsize ${wsize}
# done