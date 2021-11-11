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


from  toolbox import *

#%% Paramters for execution

path_params = "metrics/dict_parameters"
with open(path_params + ".json", "r") as json_file:
    dictionary_params = json.load(json_file)
train_times = {
    model_name: None for model_name in dictionary_params["classifiers_names"]
}
test_times = {model_name: None for model_name in dictionary_params["classifiers_names"]}
repos_cv = dictionary_params["shape_2_evolution"]
len_samp_size = dictionary_params["shape_2_all_sample_sizes"]
save_times_rewrite = {"text_dict": 1, "csv": 1, "json": 0}
save_methods = {"text_dict": 1, "csv": 1, "json": 0}



#%% Load data from CC18 data set suite
if dictionary_params["reload_data"] or "dataset_name" not in locals():
    X_data_list, y_data_list, dataset_name = load_cc18()
path_best_parameters = "metrics/cc18_all_parameters_new"

if "best_params_dict" not in locals():
    best_params_dict = open_data(path_best_parameters, "text_dict")

dataset_indices = [i for i in range(dictionary_params["dataset_indices_max"])]

# Import pretuned hyperparameters
all_params = read_params_txt("metrics/cc18_all_parameters.txt")


# Empty arrays to index times into
train_test_times = {
    metric: {model_name: {} for model_name in best_params_dict.keys()}
    for metric in ["train", "test"]
}

# For each dataset, determine wall times at each sample size
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > dictionary_params["max_shape_to_run"]:
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
        for i in range(len_samp_size):
            training_sample_sizes.append(round(np.power(10, temp + i * t)))

        ss_inds = random_sample_new(X_train, training_sample_sizes)

        # Iterate through each sample size per dataset
        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):

            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            for model_name in best_params_dict.keys():
                # parameters = create_parameters(model_name,varargin)
                model = model_define(model_name, best_params_dict, dataset)
                train_time, test_time, y_pred = calculate_time(
                    model, X_train_new, y_train_new, X_test
                )
                train_times[model_name] = train_time
                test_times[model_name] = test_time
                if (
                    sample_size_index
                    not in train_test_times["train"][model_name].keys()
                ):
                    train_test_times["train"][model_name][sample_size_index] = np.zeros(
                        repos_cv
                    )
                    train_test_times["test"][model_name][sample_size_index] = np.zeros(
                        repos_cv
                    )
                train_test_times["train"][model_name][sample_size_index][
                    k_index
                ] = train_time
                train_test_times["test"][model_name][sample_size_index][
                    k_index
                ] = test_time

        k_index += 1


# Save results as txt files
save_best_parameters(
    save_methods=save_methods,
    save_methods_rewrite=save_times_rewrite,
    path_save="results/times",
    best_parameters=train_test_times,
)
