import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

from metrics import get_performance_case

# if __name__ == '__main__':

# 	case = 'Case4'
# 	version = '_v3'
# 	bfname = f'10K_{case}'
# 	n_streams = 15
# 	n_experiments = 30
# 	window_size = 1000
# 	non_data_attr = 2
	
# 	cwd = os.getcwd()
# 	gt_folder = f"../../../../OutlierGen/exos/{case}{version}"
# 	gt_folder = os.path.join(cwd, gt_folder)
# 	print(f'gt_folder is {gt_folder}')
	
# 	performance_folder = f'pickles/performance/{case}{version}'

# 	rel_path = f'pickles/{case}{version}'

# 	df = get_performance_case(n_streams = n_streams,
# 							  bfname = bfname, 
# 							  gt_folder = gt_folder,
# 							  rel_path = rel_path,
# 							  performance_folder = performance_folder,
# 							  n_experiments= n_experiments,
# 							  window_size=window_size,
# 							  non_data_attr=non_data_attr,
# 							  vcase=case)


if __name__ == '__main__':

	case = 'Case1'
	bfname = f'w1K_{case}'
	n_streams = 2
	n_experiments = 1
	window_size = 1000
	non_data_attr = 2
	
	cwd = os.getcwd()
	gt_folder = f"../../../../OutlierGen/exos/small_cases/{case}"
	gt_folder = os.path.join(cwd, gt_folder)
	print(f'gt_folder is {gt_folder}')
	
	performance_folder = f'pickles/performance/small_cases/{case}'

	rel_path = f'pickles/small_cases/{case}'

	df = get_performance_case(n_streams = n_streams,
							  bfname = bfname, 
							  gt_folder = gt_folder,
							  rel_path = rel_path,
							  performance_folder = performance_folder,
							  n_experiments= n_experiments,
							  window_size=window_size,
							  non_data_attr=non_data_attr,
							  vcase=case)
