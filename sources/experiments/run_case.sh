#!/bin/bash
N=15
case="Case4"
D="/home/epanjei/Codes/OutlierGen/exos/cases/${case}"
bfname="100K_${case}"
relpath="pickles/cases/${case}"

for ((i=5; i<11; i++))
do
	python3 nstreams.py --ex_number $i --nstreams $N --bfname $bfname --dfolder $D --relpath $relpath
done