"""
Author: Michael Ainsworth
"""

#%% Imports 

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import ast
import openml
import time
import json 

from basic_functions_script import *
from save_hyperparameters import *
path_params = 'metrics/dict_parameters'
with open(path_params+".json",'r') as json_file:
    dictionary_params = json.load(json_file)
train_times = {model_name:None for model_name in dictionary_params['classifiers_names']}
test_times = {model_name:None for model_name in dictionary_params['classifiers_names']}
shape_2_evolution = dictionary_params['shape_2_evolution'] 
shape_2_all_sample_sizes = dictionary_params['shape_2_all_sample_sizes']
save_times_rewrite={'text_dict':0,'csv':0,'json':0}
#%% Functions


def read_params_txt(filename):
    """
    Read and parse optimized parameters from text file
    """
    params = []
    f = open(filename, "r").read()
    f = f.split("\n")
    f = f[:-1]
    for ind, i in enumerate(f):
        temp = ast.literal_eval(f[ind])
        params.append(temp)
    return params


def random_sample_new(data, training_sample_sizes):
    """
    Given X_data and a list of training sample size, randomly sample indices to be used.
    Larger sample sizes include all indices from smaller sample sizes.
    """
    temp_inds = []

    ordered = [i for i in range(len(data))]
    minus = 0
    for ss in range(len(training_sample_sizes)):
        x = sorted(sample(ordered, training_sample_sizes[ss] - minus))
        minus += len(x)
        temp_inds.append(x)
        ordered = list(set(ordered) - set(x))

    final_inds = []
    temp = []

    for i in range(len(temp_inds)):
        cur = temp_inds[i]
        final_inds.append(sorted(cur + temp))
        temp = sorted(cur + temp)

    return final_inds


def sample_large_datasets(X_data, y_data):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, 10000))
    return X_data[fin], y_data[fin]

def calculate_time(model,X_train,y_train, X_test):
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    train_time = end_time - start_time
    
    
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()
    test_time = end_time - start_time
            
    return train_time,test_time, y_pred
            
            
# Load data from CC18 data set suite
if (dictionary_params['reload_data'] or 'dataset_name' not in locals()):
    X_data_list, y_data_list, dataset_name = load_cc18()
if 'best_params_dict' not in locals():
    best_params_dict = read_params_dict_txt(path,file_type_to_load)
    
dataset_indices = [i for i in range(dictionary_params['dataset_indices_max'])]

# Import pretuned hyperparameters
all_params = read_params_txt("metrics/cc18_all_parameters.txt")


# Empty arrays to index times into
train_test_times = {metric:
    {model_name:np.zeros((shape_2_all_sample_sizes * len(dataset_indices), shape_2_evolution)) for model_name in best_params_dict.keys()} for metric in ['train','test']}

# For each dataset, determine wall times at each sample size
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > dictionary_params['max_shape_to_run']:
        X, y = sample_large_datasets(X, y)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Implement stratified 5-fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    k_index = 0
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Generate training sample sizes, logorithmically spaced
        temp = np.log10((len(np.unique(y))) * 5)
        t = (np.log10(X_train.shape[0]) - temp) / 7
        training_sample_sizes = []
        for i in range(shape_2_all_sample_sizes):
            training_sample_sizes.append(round(np.power(10, temp + i * t)))

        ss_inds = random_sample_new(X_train, training_sample_sizes)

        # Iterate through each sample size per dataset
        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):

            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]
            
            for model_name in best_params_dict.keys():
                #parameters = create_parameters(model_name,varargin)
                model = model_define(model_name,best_params_dict,dataset)
                train_time, test_time, y_pred = calculate_time(model,X_train_new,y_train_new, X_test)
                train_times[model_name] = train_time
                test_times[model_name] = test_time
                train_test_times['train'][model_name][sample_size_index + shape_2_all_sample_sizes * dataset_index][k_index] = train_time
                train_test_times['test'][model_name][sample_size_index + shape_2_all_sample_sizes * dataset_index][k_index] = test_time


        k_index += 1


# Save results as txt files
save_best_parameters(save_methods = 'text_dict',save_methods_rewrite = save_times_rewrite ,path_save = "results/times" ,best_parameters= train_test_times )
        
#np.savetxt("results/cc18_rf_times_train.txt", rf_times_train)
#np.savetxt("results/cc18_rf_times_test.txt", rf_times_test)
#np.savetxt("results/cc18_dn_times_train.txt", dn_times_train)
#np.savetxt("results/cc18_dn_times_test.txt", dn_times_test)
