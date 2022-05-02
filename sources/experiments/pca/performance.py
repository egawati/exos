import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

from metrics import get_performance_df_v2

if __name__ == '__main__':

	case = 'Case4'
	bfname = f'100K_{case}'
	nstreams = 15
	ptype = 'cases'
	df = get_performance_df_v2(n_streams=nstreams, 
	                          bfname = bfname, 
	                          gt_folder = f'/home/epanjei/Codes/OutlierGen/exos/{ptype}',
	                          rel_path =  f'pickles/{ptype}/{case}',
	                          performance_folder = f'pickles/performance/{ptype}',
	                          n_experiments=10,
	                          window_size=1000,
	                          non_data_attr=2,
	                          vcase=case)
	df