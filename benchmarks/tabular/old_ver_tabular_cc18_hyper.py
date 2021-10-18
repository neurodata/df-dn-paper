# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:17:50 2021

@author: noga mudrik
"""

"""
Author: Michael Ainsworth
"""

#%% Define executrion variables (to save memory & execution time)
reload_data = False # indicator of whether to upload the data again
nodes_combination = [20, 100, 180, 260, 340, 400] # default is [20, 100, 180, 260, 340, 400]
dataset_indices_max=72 
max_shape_to_run=10000
alpha_range_nn=[0.0001, 0.001, 0.01, 0.1]
#%% function to return to default values
def return_to_default():
    nodes_combination = [20, 100, 180, 260, 340, 400]
    dataset_indices_max = 72 
    max_shape_to_run = 10000
    alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
    return nodes_combination,dataset_indices_max,max_shape_to_run

#%% function so save cc18_all_parameters file

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import openml
import itertools


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


def sample_large_datasets(X_data, y_data):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, 10000))
    return X_data[fin], y_data[fin]


"""
Organize the data
"""

# Load data from CC18 data set suite
if (reload_data or 'task_id' not in locals()): # Load the data only if required (by reload_data or if it is not defined)
    X_data_list, y_data_list, dataset_name = load_cc18()


# Empty list to record optimal parameters
all_parameters = []


# Choose dataset indices
dataset_indices = [i for i in range(dataset_indices_max)]


"""
Deep Neural Network
"""
# Generate all combinations of nodes to tune over
test_list = nodes_combination;
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer

# For each dataset, use randomized hyperparameter search to optimize parameters
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)
        
    # Standart Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Deep network hyperparameters
    parameters = {"hidden_layer_sizes": node_range, "alpha": alpha_range_nn}

    # Random forest hyperparameters
    p = X.shape[1]
    l = list(
        set([round(p / 4), round(np.sqrt(p)), round(p / 3), round(p / 1.5), round(p)])
    )
    parameters_rf = {"max_features": l}

    mlp = MLPClassifier(max_iter=200)
    clf = RandomizedSearchCV(mlp, parameters, n_jobs=-1, cv=None, verbose=1)
    clf.fit(X, y)

    rf = RandomForestClassifier(n_estimators=500)
    clfrf = RandomizedSearchCV(rf, parameters_rf, n_jobs=-1, verbose=1)
    clfrf.fit(X, y)

    allparams = clf.cv_results_["params"]
    allparamsrf = clfrf.cv_results_["params"]

    best_params = clf.best_params_
    best_paramsrf = clfrf.best_params_

    all_parameters.append([best_params, best_paramsrf])


# Save optimal parameters to txt file
with open("metrics/cc18_all_parameters.txt", "w") as f:
    for item in all_parameters:
        f.write("%s\n" % item)
