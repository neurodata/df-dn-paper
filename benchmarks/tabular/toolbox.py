# -*- coding: utf-8 -*-

"""
Created on Tue Nov  2 05:08:11 2021

@author: noga mudrik
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import ast
import openml
import time
import json
import pandas as pd
import seaborn as sns
from os.path import exists
from tqdm import tqdm

#%% Functions to compare to original text files
def open_data(path,format_file):
    if format_file=='text_dict':
        file = open(path+".txt", "r")
        contents = file.read()

        dictionary = ast.literal_eval(contents)
        return dictionary
    


def addi(total_len_sizes_and_data = [] , path1 = 'results/cc18_rf_kappa', path2 = 'metrics/cc18_sample_sizes_new',name_model = 'RF',name_metric = 'cohen_kappa'):
    file_sizes = open(path1+".txt", "r")
    cont_sizes = file_sizes.read()
    d = np.array(cont_sizes.split('\n'))
    d_2d = [[d_el.split(' ')] for d_el in d]
    dict_to_store = {}
    len_sample_sizes = 8;
    for iterat_num, iterat in enumerate(range(0,total_len_sizes_and_data,len_sample_sizes)):
        if iterat_num < 20:
            dict_to_store[iterat_num] = {}; curr_d = d_2d[iterat:iterat+len_sample_sizes]; key_iterat = list(full_dict[iterat_num].keys())
            for row_for_sample_size in range(len_sample_sizes):
                if len(key_iterat) > row_for_sample_size:
                    curr_key = key_iterat[row_for_sample_size]; dict_to_store[iterat_num][curr_key] = tuple(curr_d[row_for_sample_size][0])
                    
    return dict_to_store


# Times functions

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


def calculate_time(model, X_train, y_train, X_test):
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_time = time.perf_counter()
    train_time = end_time - start_time

    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()
    test_time = end_time - start_time

    return train_time, test_time, y_pred


#%% Save Hyperparameters    
def save_best_parameters(
    save_methods, save_methods_rewrite, path_save, best_parameters
):
    if save_methods["text_dict"]:
        if (
            os.path.exists(path_save + ".txt")
            and save_methods_rewrite["text_dict"] == 0
        ):
            file = open(path_save + ".txt", "r")
            contents = file.read()

            dictionary = ast.literal_eval(contents)
            best_parameters_to_save = {
                **dictionary,
                **best_parameters,
            }  # This will overwrite existing models but not models that were removed
        else:
            best_parameters_to_save = best_parameters
        with open(path_save + ".txt", "w") as f:
            f.write("%s\n" % best_parameters_to_save)

    if save_methods["json"]:
        if os.path.exists(path_save + ".json") and save_methods_rewrite["json"] == 0:
            with open(path_save + ".json", "r") as json_file:
                dictionary = json.load(json_file)
            best_parameters_to_save = {**dictionary, **best_parameters}
        else:
            best_parameters_to_save = best_parameters
        with open(path_save + ".json", "w") as fp:
            json.dump(best_parameters_to_save, fp)
    if save_methods["csv"]:
        df_new_data = pd.DataFrame(best_parameters_to_save)
        if os.path.exists(path_save + ".csv") and save_methods_rewrite["csv"] == 0:
            df_old = pd.read_csv(path_save + ".csv", index=False)
            df_to_save = pd.concat([df_new_data, df_old], 1, ignore_index=True)
        else:
            df_to_save = df_new_data
        df_to_save.to_csv(path_save + ".csv", index=False)


def open_data(path, format_file):
    if format_file == "text_dict":
        file = open(path + ".txt", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        return dictionary

#%% # Functions to calculate model performance and parameters.


def create_parameters(model_name, varargin, p=None):
    if model_name == "DN":
        parameters = {
            "hidden_layer_sizes": varargin["node_range"],
            "alpha": varargin["alpha_range_nn"],
        }
    elif model_name == "RF":

        parameters = {
            "max_features": list(
                set(
                    [
                        round(p / 4),
                        round(np.sqrt(p)),
                        round(p / 3),
                        round(p / 1.5),
                        round(p),
                    ]
                )
            )
        }
    elif model_name == "GBDT":
        parameters = {
            "learning_rate": varargin["alpha_range_nn"],
            "subsample": varargin["subsample"],
        }
    else:
        raise ValueError(
            "Model name is invalid. Please check the keys of models_to_run"
        )
    return parameters


def do_calcs_per_model(
    all_parameters,
    best_parameters,
    all_params,
    model_name,
    varargin,
    varCV,
    classifiers,
    X,
    y,
    dataset_index,
    p=None,
):
    model = classifiers[model_name]
    varCVmodel = varCV[model_name]
    parameters = create_parameters(model_name, varargin, p)
    clf = RandomizedSearchCV(
        model,
        parameters,
        n_jobs=varCVmodel["n_jobs"],
        cv=varCVmodel["cv"],
        verbose=varCVmodel["verbose"],
    )
    clf.fit(X, y)
    all_parameters[model_name][dataset_index] = parameters
    best_parameters[model_name][dataset_index] = clf.best_params_
    all_params[model_name][dataset_index] = clf.cv_results_["params"]
    return all_parameters, best_parameters, all_params


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


def sample_large_datasets(X_data, y_data, max_shape_to_run=10000):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, max_shape_to_run))
    return X_data[fin], y_data[fin]


#%% Open save files
def open_dictionary_best_params(path="metrics/cc18_all_parameters_try.txt"):

    file = open(path, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    return dictionary


#%% function to return to default values
def return_to_default():
    nodes_combination = [20, 100, 180, 260, 340, 400]
    dataset_indices_max = 72
    max_shape_to_run = 10000
    alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
    subsample = [0.5, 0.8, 1.0]
    models_to_run = {"RF": 1, "DN": 1, "GBDT": 1}
    return (
        nodes_combination,
        dataset_indices_max,
        max_shape_to_run,
        models_to_run,
        subsample,
        alpha_range_nn,
    )


#%% Function to save variables to dict for later use
def save_vars_to_dict(
    classifiers,
    varargin,
    reload_data=False,
    nodes_combination=[20],
    dataset_indices_max=2,
    max_shape_to_run=10000,
    alpha_range_nn=[0.1],
    subsample=[1.0],
    path_to_save="metrics/dict_parameters.json",
    shape_2_evolution=5,
    shape_2_all_sample_sizes=8,
):
    dict_to_save = {
        "reload_data": reload_data,
        "nodes_combination": nodes_combination,
        "dataset_indices_max": dataset_indices_max,
        "max_shape_to_run": max_shape_to_run,
        "alpha_range_nn": alpha_range_nn,
        "subsample": subsample,
        "classifiers_names": tuple(classifiers.keys()),
        "varargin": varargin,
        "shape_2_evolution": shape_2_evolution,
        "shape_2_all_sample_sizes": shape_2_all_sample_sizes,
    }

    with open(path_to_save, "w") as fp:
        json.dump(dict_to_save, fp)


def model_define(model_name, best_params_dict, dataset):
    if model_name == "RF":
        model = RandomForestClassifier(
            **best_params_dict[model_name][dataset], n_estimators=500, n_jobs=-1
        )
    elif model_name == "DN":
        model = MLPClassifier(**best_params_dict[model_name][dataset])
    elif model_name == "GBDT":
        model = GradientBoostingClassifier(
            **best_params_dict[model_name][dataset], n_estimators=500
        )
    else:
        raise ValueError("Invalid Model Name")
    return model


def mod_dict(res_dict, type_to_compare):
    tot_dict = {}
    for model_name in res_dict.keys():
        modified_subdict = {}
        for dataset in res_dict[model_name].keys():
            data_set_dict = res_dict[model_name][dataset]
            data_set_dict = {
                key: val
                for key, val in data_set_dict.items()
                if isinstance(val, type_to_compare)
            }
            modified_subdict[dataset] = data_set_dict
        tot_dict[model_name] = modified_subdict
    return tot_dict
        

#%%  Kappa ECE



def read_params_dict_txt(
    path="metrics/cc18_all_parameters", type_file=".txt"
):  # Path should not include the file type (like .txt)
    """
    Read optimized parameters as saved in a dict
    """

    if type_file == ".txt":
        file = open(path + ".txt", "r")
        contents = file.read()
        best_params_dict = ast.literal_eval(contents)

    elif type_file == ".csv":
        df_old = pd.read_csv(path + ".csv", index=False)
        best_params_dict = df_old.to_dict()

    elif type_file == ".json":

        with open(path + ".json", "r") as json_file:
            best_params_dict = json.load(json_file)

    else:
        raise NameError('Invalid type file in "type_files" argument: %s' % type_file)
    return best_params_dict



def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    """
    Return ece score

    Function borrowed from: https://github.com/neurodata/kdg/blob/main/kdg/utils.py
    """

    bin_size = 1 / num_bins
    total_sample = len(true_label)
    posteriors = predicted_posterior.max(axis=1)

    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (posteriors > bin * bin_size) & (posteriors <= (bin + 1) * bin_size)
        )[0]

        acc = (
            np.nan_to_num(np.mean(predicted_label[indx] == true_label[indx]))
            if indx.size != 0
            else 0
        )
        conf = np.nan_to_num(np.mean(posteriors[indx])) if indx.size != 0 else 0
        score += len(indx) * np.abs(acc - conf)

    score /= total_sample
    return score


