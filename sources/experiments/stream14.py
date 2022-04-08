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
    
    n_streams = 2
    size = 1000
    window_size = 500
    n_attrs = 5
    sources = list()
    attributes = list()
    feature_names = {}
    counter = 0
    for i in range(n_streams):
        X = np.random.uniform(low=5, high=20, size=(size,n_attrs))
        y = np.random.randint(2, size=size)
        ts = TemporalDataStream(X, y)
        sources.append(ts)
        feature_names[i] = [f'A{j}' for j in range(n_attrs)]
        attributes.append(counter)
        counter = counter + X.shape[1]
    d = n_streams * n_attrs
    k = d
    print(f'attributes {attributes}')
    print(f'source len {len(sources)}')

    results = run_exos_simulator(sources, d, k, attributes, feature_names, 
                                 window_size, n_clusters = (), n_init_data = (), 
                                 multiplier = 10, round_flag=True)

    filename = f'{n_streams}_{size}_{window_size}_{n_attrs}.pkl'
    exos_file = open(filename, 'ab')
    pickle.dump(results, exos_file)

