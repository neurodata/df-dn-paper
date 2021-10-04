# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:43:39 2021

@author: noga mudrik
"""
import ast
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import openml
import itertools
import pandas as pd
import json
from tqdm import tqdm
# Functions to calculate model performance and parameters.

def create_parameters(model_name,varargin,p=None):
    if model_name=='DN':
        parameters = {"hidden_layer_sizes": varargin['node_range'], "alpha": varargin['alpha_range_nn']}
    elif model_name=='RF':
        
        parameters = {"max_features": list(set([round(p / 4), round(np.sqrt(p)), round(p / 3), round(p / 1.5), round(p)]))}
    elif model_name=='GBDT':
        parameters = {'learning_rate':varargin['alpha_range_nn'],'subsample':varargin['subsample']}
    else:
        raise ValueError("Model name is invalid. Please check the keys of models_to_run")
    return parameters
        
def do_calcs_per_model(all_parameters,best_parameters, all_params,model_name,varargin,varCV,classifiers,X,y,dataset_index):
    model = classifiers[model_name]
    varCVmodel = varCV[model_name]
    parameters = create_parameters(model_name,varargin)
    clf = RandomizedSearchCV(model, parameters, n_jobs=varCVmodel['n_jobs'], cv=varCVmodel['cv'], verbose=varCVmodel['verbose'])
    clf.fit(X, y)
    all_parameters[model_name][dataset_index] = parameters
    best_parameters[model_name][dataset_index] = clf.best_params_
    all_params[model_name][dataset_index] = clf.cv_results_["params"]
    return all_parameters, best_parameters,all_params

def load_cc18():
    """
    Import datasets from OpenML-CC18 dataset suite
    """
    X_data_list = []
    y_data_list = []
    dataset_name = []

    for task_num, task_id in enumerate(
        tqdm(openml.study.get_suite("OpenML-CC18").tasks)
    ):
        try:
            successfully_loaded = True
            dataset = openml.datasets.get_dataset(
                openml.tasks.get_task(task_id).dataset_id
            )
            dataset_name.append(dataset.name)
            X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )
            _, y = np.unique(y, return_inverse=True)
            X = np.nan_to_num(X)
        except TypeError:
            successfully_loaded = False
        if successfully_loaded and np.shape(X)[1] > 0:
            X_data_list.append(X)
            y_data_list.append(y)

    return X_data_list, y_data_list, dataset_name


def sample_large_datasets(X_data, y_data,max_shape_to_run=10000):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds,max_shape_to_run ))
    return X_data[fin], y_data[fin]


#%% Open save files
def open_dictionary_best_params(path="metrics/cc18_all_parameters_try.txt"):

    file = open(path, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    return dictionary
