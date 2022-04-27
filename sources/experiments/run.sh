#!/bin/bash
N=35
D="/home/epanjei/Codes/OutlierGen/exos/nstreams/${N}"

for ((i=1; i<11; i++))
do
	python3 nstreams.py --ex_number $i --nstreams $N --bfname 100K_Case1 --dfolder $D --relpath pickles/nstreams
done