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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import ast
import openml
import time
import json
import pandas as pd
import seaborn as sns
from os.path import exists
from tqdm import tqdm
from collections import Counter
import warnings

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


def random_sample_new_old(data, training_sample_sizes):
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
        cur = temp_inds[i] # cur is the current addition of indices
        final_inds.append(sorted(cur + temp))
        temp = sorted(cur + temp) # temp is the cumulative list of indices
    # final_inds is a list of lists of indices
    return final_inds


#partitions = np.array_split(np.array(range(samples)), num_classes)
occurences = dict(Counter(y_train))
# Obtain only train images and labels for selected classes
i = 0
for cls in classes:
    class_idx = np.argwhere(train_labels == cls).flatten()
    np.random.shuffle(class_idx)
    curr_indices = class_idx[: len(partitions[i])]
    final_inds.append(curr_indices)        
    
#        image_ls.append(class_img)
#        label_ls.append(np.repeat(cls, len(partitions[i])))
    i += 1
    
def random_sample_new(X_train, y_train, training_sample_sizes, classes = 10, seed_rand = 0, min_rep_per_class = 1):
    """
    Peforms multiclass predictions for a random forest classifier
    with fixed total samples
    """
    
    random.seed(seed_rand)
    np.random.seed(seed_rand)
    training_sample_sizes = sorted(training_sample_sizes)
    if isinstance(classes,(int,float) ):
        classes_all = np.unique(y_train)
        classes_spec = np.random.choice(classes_all,10)
        num_classes = classes
    elif isinstance(classes, (tuple,list, numpy.ndarray)):
        classes_spec = classes
        num_classes = len(classes)
    else:
        raise TypeError('Unrecognized classes type: %s'%type(classes))
    if num_classes > len(np.unique(y_train)):
        warnings.warn("Number of required classes is higher than possible. Num. of classes was re-defined as the number of unique classes")
        num_classes = len(np.unique(y_train))
        classes_spec = classes_spec[:num_classes]
       
    basic_indices = np.argwhere(np.array(y_train)>1).T[0]
    previous_partitions_len = np.zeros(num_classes)
    previous_inds = {class_val:[] for class_val in clasees_spec}
    prev_samp_size = 0
    final_inds = []
    if np.floor(np.min(samp_size)/num_classes) < min_rep_per_class:
        warnings.warn("Not enough samples for each class, decreasing number of classes")
        num_classes = np.floor(np.min(samp_size)/min_rep_per_class)
        classes_spec = classes_spec[:num_classes]
    for samp_size_count, samp_size in enumerate(training_sample_sizes):
        #real_samp_size = samp_size - prev_samp_size
        partitions = np.array_split(np.array(range(samp_size)), num_classes)
        
        # partition real include the number of additional indices we need to find
        partitions_real = [len(part_new)-previous_partitions_len[class_c] for class_c,part_new in partitions ]
        if samp_size <= len(basic_indices):
            indices_classes_addition_all = [] # For each samples size = what indices are taken for all classes together
            for class_count, class_val in enumerate(classes_spec):
                indices_class = np.argwhere(np.array(y_train) == class_val).T[0]
                indices_class_original = [ind_class for ind_class in indices_class if ind_class not in previous_inds[class_val]]
                np.random.shuffle(indices_class_original)
                if  partitions_real[class_count] <= len(indices_class_original):
                    indices_class_addition = indices_class_original[:partitions_real[class_count] ]
                    previous_inds[class_val].extend(indices_class_addition)
                    indices_classes_addition_all.extend(indices_class_addition)
                
                else:
                    raise ValueError('Class %s does not have enough samples'%str(class_val))
            if final_inds:
                indices_prev = final_inds[-1].copy()
            else:
                indices_prev = []
                
            indices_class_addtion_and_prev = indices_prev + indices_class_addition
            final_inds.append(indices_class_addtion_and_prev)
            previous_partitions_len = [len(parti) for parti in partitions]    
            prev_samp_size = samp_size
                
        else:
            raise ValueError('Samp size is too high given the data')     
        
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


