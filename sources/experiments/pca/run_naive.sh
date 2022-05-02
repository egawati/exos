#!/bin/bash
N=50
D="/home/epanjei/Codes/OutlierGen/exos/nstreams/${N}"
relpath="pickles/nstreams"
bfname="100K_Case1"

for ((i=1; i<2; i++))
do
	python3 nstreams_naive_pca.py --ex_number $i --nstreams $N --bfname $bfname --dfolder $D --relpath $relpath
done