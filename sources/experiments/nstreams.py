import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream

from multiprocessing import set_start_method

from exos.streams import run_exos_simulator

import pickle

if __name__ == '__main__':
    set_start_method("spawn")

    profiling = True
    round_flag = False
    multiplier = 10
    threshold=0.05
    
    n_streams = 32
    window_size = 1000

    folder = '/home/epanjei/Codes/OutlierGen/exos/nstreams/thirtytwo'
    dest_folder = '/home/epanjei/Codes/EXOS/sources/experiments/pickles/nstreams'
    basic_filename = '32_100K_Case1'
    filenames = [f'{i}_{basic_filename}.pkl' for i in range(n_streams)]

    F = list() ## storing outlying attributes ground truth
    labels = list() ## storing info whether a data point is an outlier (1) or inlier (0)

    sources = list()
    attributes = list()
    feature_names = {} 
    
    d = 0

    for i in range(n_streams):
        df = pd.read_pickle(f'{folder}/{filenames[i]}')
        F.append(df['outlying_attributes'])
        y = np.array(df['label'])
        labels.append(y)
        df = df.drop(['label', 'outlying_attributes'], axis=1)
        X = df.to_numpy()

        columns = list(df.columns)
        feature_names[i] = columns
        attributes.append(d)
        #counter += len(columns)
        d += len(columns)
        ts = TemporalDataStream(X,y, ordered=True)
        sources.append(ts)

    k = 1 ## number of principal component used

    print(f'attributes {attributes}')
    print(f'source len {len(sources)}')
    print(f'total attributes {d}')

    results = run_exos_simulator(sources, d, k, attributes, feature_names, 
                                 window_size, n_clusters = (), n_init_data = (), 
                                 round_flag=round_flag, threshold=threshold)

    filepath = f'{dest_folder}/{basic_filename}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    