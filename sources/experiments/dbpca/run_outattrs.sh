#!/bin/bash

M=2
bname="w1K"
nsets=1
wsize=1000
mu=5
sigma=0.5
max_noutattrs=10
version=""
versionp="v"
non_data_attr=2


data_dir="../../../../OutlierGen/exos/small_cases"
gt_folder="../../../../OutlierGen/exos/small_cases"
relpath="pickles/small_cases"
performance_folder="pickles/performance/small_cases"


case="Case4"
D="${data_dir}/${case}${version}"
bfname="${bname}_${case}"
path="${relpath}/${case}${version}"
threshold=0.1
reducer=0.005

for ((i=1; i<$max_noutattrs+1; i++))
do
	python3 outattrs.py --ex_number $nsets --nstreams $M --bfname $bfname --dfolder $D --relpath $path --wsize ${wsize} --init_mu ${mu} --init_sigma ${sigma} --noutattrs ${i} --threshold ${threshold}
	threshold=$( bc <<<"$threshold - $reducer" )
done

for ((i=1; i<$max_noutattrs+1; i++))
do
	python3 performance_outattrs.py --bfname ${bname} --case ${case} --relpath ${relpath} --gt_folder ${gt_folder} --performance_folder ${performance_folder} --nstreams $M --nsets ${nsets} --noutattrs ${i} --window_size ${wsize} --version ${versionp} --non_data_attr ${non_data_attr}
done


case="Case1"
D="${data_dir}/${case}${version}"
bfname="${bname}_${case}"
path="${relpath}/${case}${version}"
threshold=0.1
for ((i=1; i<$max_noutattrs+1; i++))
do
	python3 outattrs.py --ex_number $nsets --nstreams $M --bfname $bfname --dfolder $D --relpath $path --wsize ${wsize} --init_mu ${mu} --init_sigma ${sigma} --noutattrs ${i} --threshold ${threshold}
	threshold=$( bc <<<"$threshold - $reducer" )
done

for ((i=1; i<$max_noutattrs+1; i++))
do
	python3 performance_outattrs.py --bfname ${bname} --case ${case} --relpath ${relpath} --gt_folder ${gt_folder} --performance_folder ${performance_folder} --nstreams $M --nsets ${nsets} --noutattrs ${i} --window_size ${wsize} --version ${versionp} --non_data_attr ${non_data_attr}
done
