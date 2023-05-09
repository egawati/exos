import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream

from multiprocessing import set_start_method

from exos.streams import run_exos_simulator

import pickle

import os

def run_experiments(case, 
                    window_size = 1000,
                    n_streams=4,
                    relpath='Results/exos/Position', 
                    data_folder='../Datasets/ConceptDrift/WindowChanges/Position'):
    cwd = os.getcwd()
    round_flag = False
    multiplier = 5
    threshold= 0.0
    nclusters = 1
    init_mu = 5
    init_sigma = 0.5
    normalized = True
    version = None

    k = 1             ## number of principal component used 
    
    basic_filename = f'{case}'

    data_folder = os.path.join(cwd, data_folder)
    print(f'Data folder is {data_folder}')

    bfnames = list()
    for i in range(n_streams):
        bfnames.append(f'{i}_{case}')

    n_streams = len(bfnames)
    data_filetype = 'pkl'

    filenames = [f'{bfname}.{data_filetype}' for bfname in bfnames]

    if version is not None:
        filenames = [f'{version}_{filename}' for filename in filenames]
        basic_filename = f'{version}_{case}'


    F = list() ## storing outlying attributes ground truth
    labels = list() ## storing info whether a data point is an outlier (1) or inlier (0)

    sources = list()
    attributes = list()
    feature_names = {} 
    
    d = 0
    n_init_centroids = list()
    
    for i in range(n_streams):
        df = None
        if data_filetype == 'csv':
            df = pd.read_csv(f'{data_folder}/{filenames[i]}', sep=',')
        elif data_filetype == 'pkl':
            df = pd.read_pickle(f'{data_folder}/{filenames[i]}')
        F.append(df['outlying_attributes'])
        y = np.array(df['label'])
        labels.append(y)
        df = df.drop(['label', 'outlying_attributes'], axis=1)
        X = df.to_numpy()
        n_init_centroids.append(X[0:nclusters,:])
        columns = list(df.columns)
        feature_names[i] = columns
        attributes.append(d)
        #counter += len(columns)
        d += len(columns)
        ts = TemporalDataStream(X,y, ordered=True)
        sources.append(ts)
 

    print(f'attributes {attributes}')
    print(f'source len {len(sources)}')
    print(f'total attributes {d}')


    results = run_exos_simulator(sources, d, k, attributes, feature_names, 
                                 window_size, n_clusters = (nclusters,), n_init_centroids = n_init_centroids, 
                                 round_flag=round_flag, threshold=threshold, 
                                 init_mu=init_mu, init_sigma=init_sigma, normalized=normalized)
    
    dirpath = os.path.join(cwd, relpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filepath = f'{dirpath}/{basic_filename}.pkl' ##Intel.pkl
    print(f'save result to {filepath}')
    with open(filepath, 'wb') as f:
        pickle.dump(results, f) 

if __name__ == '__main__':
    set_start_method("spawn")
    bname = f'cd_pos'
    ncases = 11
    cases = [ f'{bname}_{i}' for i in range(1, ncases+1)]
    for case in cases:
        run_experiments(case=case, 
                        window_size = 1000,
                        n_streams=4,
                        relpath='Results/exos/Position', 
                        data_folder='../Datasets/ConceptDrift/WindowChanges/Position')
        print(f'Done with case {case}')



