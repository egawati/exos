import sys
import os

module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, module_path)

from metrics import get_performance_df_v2

if __name__ == '__main__':

	df = get_performance_df_v2(n_streams=5, 
	                          bfname = '100K_Case1', 
	                          gt_folder = '/home/epanjei/Codes/OutlierGen/exos/nstreams',
	                          rel_path =  'pickles/nstreams',
	                          performance_folder = 'pickles/performance/nstreams',
	                          n_experiments=10,
	                          window_size=1000,
	                          non_data_attr=2)
