import pandas as pd
import os
import pickle

def exos_running_time_df(results):
    df_dict = {}
    output = results['output']
    windows = output.keys()
    neigh_times = list()
    est_times = list()
    out_attrs_times = list()
    for window in windows:
        est_times.append(output[window]['est_time'])
        del output[window]['est_time']
        neigh_time = max([output[window][stream_id]['temporal_neighbor_time'] for stream_id in output[window].keys()])
        neigh_times.append(neigh_time)
        out_attrs_time = max([output[window][stream_id]['out_attrs_time'] for stream_id in output[window].keys()])
        out_attrs_times.append(out_attrs_time)
    df_dict['windows'] = windows
    df_dict['est_times'] = est_times
    df_dict['neigh_times'] = neigh_times
    df_dict['out_attrs_times'] = out_attrs_times
    return pd.DataFrame.from_dict(df_dict)


def unpickled_results(filename):
    result_file = open(filename, 'rb')
    results = pickle.load(result_file)
    return results

def get_confusion_matrix(out_attrs, ground_truth, total_attributes):
    TP = len(set(out_attrs) & set(ground_truth))
    FP = len(set(out_attrs) - set(ground_truth))
    FN = len(set(ground_truth) - set(out_attrs))
    TN = total_attributes - (TP + FP + FN)
    confusion_matrix = {'TP' : TP,
                        'FP' : FP,
                        'FN' : FN,
                        'TN' : TN}
    return confusion_matrix

def get_confusion_matrix_v2(out_attrs, ground_truth, total_attributes):
    TP = 0 ## increment by one when an outlying attribute is in the ground truth
    FP = 0 ## increment by one when an outlying attribute is not in the ground truth
    for out_attr in out_attrs.keys():
        if out_attr in ground_truth:
            TP = TP + 1
        else:
            FP = FP + 1
    
    FN = 0 ## an attribute that is in ground truth is not found in outlying attribute list
    for gt in ground_truth:
        if gt not in out_attrs:
            FN = FN + 1
    
    TN = total_attributes - (TP + FP + FN) ## an attribute that is not in the ground truth is neither found in outlying attribute list
    confusion_matrix = {'TP' : TP,
                        'FP' : FP,
                        'FN' : FN,
                        'TN' : TN}
    return confusion_matrix


def compute_precision(confusion_matrix):
    if confusion_matrix['TP']  == 0:
        return 0
    precision = confusion_matrix['TP'] / ( confusion_matrix['TP'] +  confusion_matrix['FP'])
    return precision

def compute_recall(confusion_matrix):
    if confusion_matrix['TP'] == 0:
        return 0
    recall = confusion_matrix['TP'] / ( confusion_matrix['TP'] +  confusion_matrix['FN'])
    return recall

def compute_f1_score(precision, recall):
    if precision+recall == 0:
        return 0
    f1_score = (2 * precision * recall ) / (precision + recall)
    return f1_score

def compute_performance_exos(gt_folder, gt_filenames, gt_filetype,
                             result_folder, result_filename,
                             n_streams, window_size, non_data_attr=2):
    """
    compute TP, FP, TN and FN and then compute precision, recall and F1 score
    """

    result_path = f'{result_folder}/{result_filename}'
    results = unpickled_results(result_path)
    windows = tuple(results['output'].keys()) ## get tuple of window ids : (window_0, window_1, ...)
    n_outliers = 0
    accuracies = {}
    print(f'window size {window_size}')
    acc_info = {}
    for i in range(n_streams):
        precision_list = list()
        recall_list = list()
        f1_score_list = list()
        gt_path = f'{gt_folder}/{gt_filenames[i]}' #ground truth filepath
        df = None
        if gt_filetype == 'pkl':
            df = pd.read_pickle(gt_path)
        else:
            df = pd.read_csv(gt_path, sep=',')

        print(f'Checking {gt_path}')

        n_attributes = df.shape[1] - non_data_attr
        df = df[['label', 'outlying_attributes']]

        stream_list = list()
        out_attr_list = list()
        gt_list = list()
        TP_list = list()
        FP_list = list()
        TN_list = list()
        FN_list = list()
        window_list = list()
        outlier_list = list()

        for j, window in enumerate(windows):
            outlier_indices = results['output'][window][i]['outlier_indices']
            # print(f'window {window}')
            # print(f'outlier_indices {outlier_indices}')
            if outlier_indices is not None:
                outlier_indices = outlier_indices[i]
                new_df = df.iloc[j*window_size:(j+1)*window_size].reset_index(drop=True)
                n_outliers += len(outlier_indices)
                ground_truth = new_df.iloc[outlier_indices].reset_index(drop=True)
                outlying_attributes = results['output'][window][i]['out_attrs']
                # print(f'outlier_indices {outlier_indices}')
                for idx , gt in ground_truth.iterrows():
                    # print(f'idx {idx} at window {window} at stream {i}')
                    # print(f'out_attrs is {outlying_attributes[idx]}')
                    # print(f'ground_truth is {gt["outlying_attributes"]}')
                    # print(f'at stream {i+1}, total attributes is {n_attributes}')
                    confusion_matrix = get_confusion_matrix_v2(outlying_attributes[idx], 
                                                            gt['outlying_attributes'], 
                                                            n_attributes)
                    precision = compute_precision(confusion_matrix)
                    recall = compute_recall(confusion_matrix)
                    f1_score = compute_f1_score(precision, recall)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_score_list.append(f1_score)

                    out_attrs = outlying_attributes[idx]
                    for key, val in out_attrs.items():
                        out_attrs[key] = round(val, 3)
                    out_attr_list.append(list(out_attrs.keys()))

                    gt_list.append(gt['outlying_attributes'])
                    TP_list.append(confusion_matrix['TP'])
                    FP_list.append(confusion_matrix['FP'])
                    FN_list.append(confusion_matrix['FN'])
                    TN_list.append(confusion_matrix['TN'])
                    stream_list.append(i+1)
                    window_list.append(window)
                outlier_list.extend(outlier_indices)

        accuracies[i] = {'precision' : precision_list,
                         'recall' : recall_list,
                         'f1_score' : f1_score_list,}
        acc_info[i] = {'stream_id' : stream_list,
                       'window' : window_list,
                       'outlier_indices': outlier_list,
                       'outlying_attributes' : out_attr_list,
                       'ground_truth' : gt_list,
                       'TP' : TP_list,
                       'FP' : FP_list,
                       'FN' : FN_list,
                       'TN' : TN_list,
                       'precision' : precision_list,
                       'recall' : recall_list,
                       'f1_score' : f1_score_list,}
    return n_outliers, accuracies, results['simulator_time'], acc_info

def aggregate_performance_exos(gt_folder, gt_filenames, gt_filetype,
                               result_folder, result_filename,
                               performance_folder,
                               n_streams, window_size, non_data_attr=2):
    n_outliers, accuracies, simulation_time, acc_info = compute_performance_exos(gt_folder, 
                                                                     gt_filenames, 
                                                                     gt_filetype,
                                                                     result_folder, 
                                                                     result_filename,
                                                                     n_streams, 
                                                                     window_size, 
                                                                     non_data_attr)
    df= pd.DataFrame(accuracies[0])
    info_df = pd.DataFrame(acc_info[0])
    for i in range(1, n_streams):
        ndf = pd.DataFrame(accuracies[i])
        df = pd.concat([df, ndf], axis=0)
        ninfo_df = pd.DataFrame(acc_info[i])
        info_df = pd.concat([info_df, ninfo_df], axis=0)
    
    ### for sanity checking
    if n_outliers == df.shape[0]:
        if not os.path.exists(performance_folder):
            os.makedirs(performance_folder)
        info_df.to_pickle(f"{performance_folder}/{result_filename}")
        print(info_df)
    return df, simulation_time

def get_performance_df_exos(case='Combed',
                            bfnames = ('AcademicBlock_Floor_1', 'AcademicBlock_Floor_2', 'AcademicBlock_Floor_3', 'AcademicBlock_Floor_4'),
                          gt_filetype = '.csv',
                          gt_folder = '../../Datasets/Combed',
                          result_filename = "Combed.pkl",
                          rel_path =  '../../Pickles/exos/dbpca/Combed/w1000',
                          performance_folder = '../../Pickles/performance/exos/dbpca/Combed/w1000',
                          window_size=1000,
                          non_data_attr=2):
    cwd = os.getcwd()
    
    n_streams = len(bfnames)

    gt_filenames = [f'{bfname}.{gt_filetype}' for bfname in bfnames]
    
    result_folder = os.path.join(cwd, rel_path)
    
    performance_folder=os.path.join(cwd, performance_folder)
    
    simulation_times = list()
    precision_means = list()
    recall_means = list()
    f1_score_means = list()
    
    df, s_time = aggregate_performance_exos(gt_folder=gt_folder, 
                                       gt_filenames=gt_filenames, 
                                       gt_filetype = gt_filetype,
                                       result_folder=result_folder, 
                                       result_filename= result_filename,
                                       performance_folder=performance_folder,
                                       n_streams=n_streams, 
                                       window_size=window_size, 
                                       non_data_attr=2)
    simulation_times.append(s_time)
    precision_means.append(df['precision'].mean())
    recall_means.append(df['recall'].mean())
    f1_score_means.append(df['f1_score'].mean())
    
    accuracy = {'window_size' : window_size, 
                'precision' : precision_means, 
                'recall': recall_means,
                'f1_score' : f1_score_means,
                'running_time' : simulation_times}

    df_aggregate = pd.DataFrame(accuracy)
    
    if not os.path.exists(performance_folder):
        os.makedirs(performance_folder)

    df_aggregate.to_pickle(f'{performance_folder}/aggregate_{result_filename}')
    result_filename = result_filename.replace('.pkl', '')
    df_aggregate.to_csv(f'{performance_folder}/aggregate_{result_filename}.csv', sep=',', index=False)
    print(f'Comparing experiments stored in {result_folder} with ground truth stores in {gt_folder}')
    print(f'Perfomance is stored is {performance_folder}')
    print(df_aggregate)
    return df_aggregate

def recap_performance_by_wsize(windows = (),
                               rel_path =  f'../../../Pickles/performance/exos/dbpca',
                               case = 'Case1',
                               version = None,
                               pkl_type = 'pkl',
                               n_streams = 5):
    avg_precision = list()
    avg_recall = list()
    avg_f1_score = list()
    avg_running_time = list()
    for window_size in windows:
        window_string = f'w{window_size}'    
        bfname = f'aggregate_{case}'
        if version is not None:
            bfname = f'{version}_{bfname}'
        file_path = f'{rel_path}/{case}/{window_string}/{bfname}.{pkl_type}'
        print(file_path)
        df = pd.read_pickle(file_path)
        avg_precision.append(df['precision'].mean())
        avg_recall.append(df['recall'].mean())
        avg_f1_score.append(df['f1_score'].mean())
        avg_running_time.append(df['running_time'].mean())

    performance = {'window' : windows, 
                   'avg_precision' : avg_precision,
                   'avg_recall' : avg_recall,
                   'avg_f1_score' : avg_f1_score,
                   'avg_running_time' : avg_running_time,}
    df = pd.DataFrame(performance)
    df.to_pickle(f'{rel_path}/avg_performance_{n_streams}_{case}_Window.pkl')
    df.to_csv(f'{rel_path}/avg_performance_{n_streams}_{case}_Window.csv')
    return df

def recap_performance_by_threshold(thresholds=(),
                               rel_path =  f'../../../Pickles/performance/exos/dbpca/Case1/w1000',
                               basic_filename='Case1_w1000',
                               version = None,
                               pkl_type = 'pkl',
                               n_streams = 5):
    avg_precision = list()
    avg_recall = list()
    avg_f1_score = list()
    avg_running_time = list()
    for threshold in thresholds:  
        bfname = f'aggregate_{basic_filename}_t{threshold}'
        if version is not None:
            bfname = f'{version}_{bfname}'
        file_path = f'{rel_path}/{bfname}.{pkl_type}'
        print(file_path)
        df = pd.read_pickle(file_path)
        avg_precision.append(df['precision'].mean())
        avg_recall.append(df['recall'].mean())
        avg_f1_score.append(df['f1_score'].mean())
        avg_running_time.append(df['running_time'].mean())

    performance = {'threshold' : thresholds,
                   'avg_precision' : avg_precision,
                   'avg_recall' : avg_recall,
                   'avg_f1_score' : avg_f1_score,
                   'avg_running_time' : avg_running_time,}
    df = pd.DataFrame(performance)
    df.to_pickle(f'{rel_path}/avg_performance_{n_streams}_{basic_filename}_Threshold.pkl')
    df.to_csv(f'{rel_path}/avg_performance_{n_streams}_{basic_filename}_Threshold.csv')
    return df

def recap_performance_by_k(ks=(),
                               rel_path =  f'../../../Pickles/performance/exos/dbpca/Case1/w1000',
                               basic_filename='Case1_w1000',
                               version = None,
                               pkl_type = 'pkl',
                               n_streams = 5):
    avg_precision = list()
    avg_recall = list()
    avg_f1_score = list()
    avg_running_time = list()
    for k in ks:  
        bfname = f'aggregate_{basic_filename}_k{k}'
        if version is not None:
            bfname = f'{version}_{bfname}'
        file_path = f'{rel_path}/{bfname}.{pkl_type}'
        print(file_path)
        df = pd.read_pickle(file_path)
        avg_precision.append(df['precision'].mean())
        avg_recall.append(df['recall'].mean())
        avg_f1_score.append(df['f1_score'].mean())
        avg_running_time.append(df['running_time'].mean())

    performance = {'k' : ks,
                   'avg_precision' : avg_precision,
                   'avg_recall' : avg_recall,
                   'avg_f1_score' : avg_f1_score,
                   'avg_running_time' : avg_running_time,}
    df = pd.DataFrame(performance)
    df.to_pickle(f'{rel_path}/avg_performance_{n_streams}_{basic_filename}_k.pkl')
    df.to_csv(f'{rel_path}/avg_performance_{n_streams}_{basic_filename}_k.csv')
    return df