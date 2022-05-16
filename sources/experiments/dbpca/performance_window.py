import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

from metrics import get_performance_window

if __name__ == '__main__':

	case = 'Case4'
	version = '_v3'
	bfname = f'10K_{case}'
	n_streams = 15
	exp_num = 1
	window_sizes = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500)
	non_data_attr = 2
	
	cwd = os.getcwd()
	gt_folder = f"../../../../OutlierGen/exos/{case}{version}"
	gt_folder = os.path.join(cwd, gt_folder)
	print(f'gt_folder is {gt_folder}')
	
	performance_folder = f'pickles/performance/{case}{version}/windows/exp_{exp_num}'

	rel_path = f'pickles/{case}{version}/windows/exp_{exp_num}'

	df = get_performance_window(n_streams= n_streams, 
	                          bfname= bfname, 
	                          gt_folder=gt_folder,
	                          rel_path=rel_path,
	                          performance_folder=performance_folder,
	                          exp_num=exp_num,
	                          window_sizes=window_sizes,
	                          non_data_attr=non_data_attr,
	                          vcase=case)
