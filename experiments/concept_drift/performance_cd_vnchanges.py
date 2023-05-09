import sys
import os
import argparse

module_path = os.path.abspath(os.path.join('../'))
sys.path.insert(1, module_path)

from metrics import get_performance_df_exos


def populate_performance_data(case, 
							  window_size=1000, 
							  n_streams=4, 
							  non_data_attr=2,
							  gt_folder='../Datasets/ConceptDrift/WindowChanges/NChanges', 
							  rel_path='Results/exos',
							  performance_folder='Perfomance/exos'):
	
	version = None
	bfnames = list()
	for i in range(n_streams):
		bfnames.append(f'{i}_{case}')

	result_filename = f"{case}.pkl"

	if version is not None:
		bfnames = [f'{version}_{bfname}' for bfname in bfnames]
		result_filename = f'{version}_{result_filename}'

	gt_filetype = 'pkl'

	cwd = os.getcwd()
	gt_folder = os.path.join(cwd, gt_folder)
	performance_folder = os.path.join(cwd, performance_folder)
	rel_path = os.path.join(cwd, rel_path)

	df = get_performance_df_exos(case = case,
							     bfnames = bfnames,
								 gt_filetype = gt_filetype,
								 gt_folder = gt_folder,
								 result_filename = result_filename,
								 rel_path =  rel_path,
								 performance_folder = performance_folder,
								 window_size=window_size,
								 non_data_attr=non_data_attr)

if __name__ == '__main__':
	# bname = f'cd_nchanges'
	# folder = 'NChanges4'
	bname = f'cd_mdist'
	folder = f'MDist2'
	ncases = 11
	cases = [ f'{bname}_{i}' for i in range(1, ncases+1)]
	for case in cases:
		populate_performance_data(case, 
			  window_size=1000, 
			  n_streams=4, 
			  non_data_attr=2,
			  gt_folder=f'../Datasets/ConceptDrift/WindowChanges/{folder}', 
			  rel_path=f'Results/exos/{folder}',
			  performance_folder=f'Perfomance/exos/{folder}')
	