import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

from metrics import get_performance_case

import argparse

def define_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfname', default='w1K', type=str)
    parser.add_argument('--case', default='Case4', type=str)
    parser.add_argument('--relpath', default='pickles/small_cases/', type=str)
    parser.add_argument('--gt_folder', default="../../../../OutlierGen/exos/small_cases", type=str)
    parser.add_argument('--performance_folder', default='pickles/performance/small_cases', type=str)
    parser.add_argument('--nstreams', default=2, type=int)
    parser.add_argument('--nsets', default=1, type=int)
    parser.add_argument('--noutattrs', default=2, type=int)
    parser.add_argument('--window_size', default=1000, type=int)
    parser.add_argument('--version', default='', type=str)
    parser.add_argument('--non_data_attr', default=2, type=int)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':

	args = define_arguments()

	case = args.case
	version = args.version
	if version=='v':
		version= ''
	bfname = f'{args.bfname}_{case}'
	n_streams = args.nstreams
	n_experiments = args.nsets
	window_size = args.window_size
	non_data_attr = args.non_data_attr
	
	cwd = os.getcwd()
	gt_folder = f"{args.gt_folder}/{case}{version}"
	gt_folder = os.path.join(cwd, gt_folder)
	print(f'gt_folder is {gt_folder}')
	
	performance_folder = f'{args.performance_folder}/{case}{version}'

	rel_path = f'{args.relpath}/{case}{version}'

	df = get_performance_case(n_streams = n_streams,
							  bfname = bfname, 
							  gt_folder = gt_folder,
							  rel_path = rel_path,
							  performance_folder = performance_folder,
							  n_experiments= n_experiments,
							  window_size=window_size,
							  non_data_attr=non_data_attr,
							  vcase=case,
							  noutattrs=args.noutattrs)
