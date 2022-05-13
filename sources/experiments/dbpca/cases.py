import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream

from multiprocessing import set_start_method

from exos.streams import run_exos_simulator

import pickle

import os
import argparse

def define_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_number', type=int, required=True)
    parser.add_argument('--nstreams', type=int, required=True)
    parser.add_argument('--bfname', type=str, required=True)
    parser.add_argument('--dfolder', type=str, required=True)
    parser.add_argument('--relpath', type=str, required=True)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--wsize', default=1000, type=int)
    parser.add_argument('--multiplier', default=10, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--round_flag', default=False, type=bool)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':

    args = define_arguments()
    set_start_method("spawn")

    ex_number = args.ex_number
    
    round_flag = args.round_flag
    multiplier = args.multiplier
    threshold= args.threshold
    n_streams = args.nstreams 
    window_size = args.wsize
    bfname = args.bfname    # example: 100K_Case1
    k = args.k              ## number of principal component used 
    rel_path = f'{args.relpath}/{ex_number}' ## example: pickles/nstreams

    cwd = os.getcwd()
    data_folder = f'{args.dfolder}/{ex_number}'
    data_folder = os.path.join(cwd, data_folder)
    print(f'Data folder is {data_folder}')

    basic_filename = f'{n_streams}_{bfname}_{ex_number}' ##15_10K_Case1_1
    filenames = [f'{i}_{basic_filename}.pkl' for i in range(n_streams)] ##0_15_10K_Case1_1.pkl

    F = list() ## storing outlying attributes ground truth
    labels = list() ## storing info whether a data point is an outlier (1) or inlier (0)

    sources = list()
    attributes = list()
    feature_names = {} 
    
    d = 0
    n_init_centroids = list()
    for i in range(n_streams):
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
                                 window_size, n_clusters = (), n_init_centroids = n_init_centroids, 
                                 round_flag=round_flag, threshold=threshold)
    
    dirpath = os.path.join(cwd, args.relpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filepath = f'{dirpath}/{basic_filename}.pkl' ##15_10K_Case1_1.pkl
    print(f'save result to {filepath}')
    with open(filepath, 'wb') as f:
        pickle.dump(results, f) 