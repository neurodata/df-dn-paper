"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Noga Mudrik
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import ast
import openml
import pandas as pd
import json

from  toolbox import *

#%%
reload_data = False  # indicator of whether to upload the data again

path_save = "metrics/cc18_all_parameters_new"
path_params = "metrics/dict_parameters"

with open(path_params + ".json", "r") as json_file:
    dictionary_params = json.load(json_file)


path = path_save  # "metrics/cc18_all_parameters"
type_file = ".txt"
dataset_indices_max = dictionary_params["dataset_indices_max"]
max_shape_to_run = dictionary_params["max_shape_to_run"]
file_type_to_load = ".txt"

# Number of repetitions for each CV fold at each sample size
reps = 5
shape_2_all_sample_sizes = dictionary_params["shape_2_all_sample_sizes"]
shape_2_evolution = dictionary_params["shape_2_evolution"]
n_splits = dictionary_params["shape_2_evolution"]

models_to_run = {"RF": 1, "DN": 1, "GBDT": 1}


#%% Create data structures


# Import CC18 data and pretuned hyperparameters
if reload_data or "dataset_name" not in locals():
    X_data_list, y_data_list, dataset_name = load_cc18()
# Upload best parameters
if "best_params_dict" not in locals():
    best_params_dict = open_data(path, "text_dict")

train_indices = [i for i in range(dataset_indices_max)]

# Create empty arrays to index sample sizes, kappa values, and ece scores
all_sample_sizes = np.zeros((len(train_indices), shape_2_all_sample_sizes))

# Empty arrays to index times into
train_times = {
    model_name: None for model_name in dictionary_params["classifiers_names"]
}
test_times = {model_name: None for model_name in dictionary_params["classifiers_names"]}

train_test_times = {model_name: {} for model_name in best_params_dict.keys()}



evolution_dict = {
    metric: {model_name: {} for model_name in best_params_dict.keys()}
    for metric in ["cohen_kappa", "ece"]
}


#%% For each dataset, train and predict for every sample size
# Record outputs using Cohen's Kappa and ECE
for dataset_index, dataset in enumerate(train_indices):
    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Implement stratified 5-fold cross validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    k_index = 0
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Generate training sample sizes, logorithmically spaced
        temp = np.log10((len(np.unique(y))) * reps)
        t = (np.log10(X_train.shape[0]) - temp) / 7
        training_sample_sizes = []
        for i in range(8):
            training_sample_sizes.append(round(np.power(10, temp + i * t)))

        ss_inds = random_sample_new(X_train, training_sample_sizes)

        # Iterate through each sample size per dataset
        for sample_size_index, real_sample_size in enumerate(training_sample_sizes):
            cohen_ece_results_dict = {
                metric: {
                    model_name: np.zeros((reps))
                    for model_name in best_params_dict.keys()
                }
                for metric in ["cohen_kappa", "ece"]
            }
            train_test_times_cur = {model_name: np.zeros((reps)) for model_name in best_params_dict.keys()    }

            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            # Repeat for number of repetitions, averaging results
            for model_name, model_best_params in best_params_dict.items():
                if models_to_run[model_name]:
                    if dataset not in train_test_times["cohen_kappa"][model_name].keys():
                        train_test_times["cohen_kappa"][model_name][dataset] = {}
                        train_test_times["ece"][model_name][dataset] = {}

                    for ii in range(reps): # ii 
                        try:
                            model = model_define(
                                model_name, best_params_dict, dataset_index
                            )
                            
                            start_time = time.perf_counter()
                            model.fit(X_train_new, y_train_new)
                            end_time = time.perf_counter()
                            train_time = end_time - start_time

                            
                            predictions = model.predict(X_test)
                            predict_probas = model.predict_proba(X_test)

                            cohen_kappa = cohen_kappa_score(y_test, predictions)
                            cohen_ece_results_dict["cohen_kappa"][model_name][
                                ii
                            ] = cohen_kappa
                            ece = get_ece(predict_probas, predictions, y_test)
                            cohen_ece_results_dict["ece"][model_name][ii] = ece
                            
  
                            train_test_times_cur[model_name][ ii ] = train_time
                            
                            
                        except ValueError:
                            print(model_name)

                    if dataset not in evolution_dict["cohen_kappa"][model_name].keys():
                        evolution_dict["cohen_kappa"][model_name][dataset] = {}
                        evolution_dict["ece"][model_name][dataset] = {}
                    if (real_sample_size not in evolution_dict["cohen_kappa"][model_name][dataset].keys()):
                        evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size] = []
                        evolution_dict["ece"][model_name][dataset][real_sample_size] = []
                        train_test_times[model_name][dataset][real_sample_size] = []
                        train_test_times[model_name][dataset][real_sample_size] = []

                    evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size].append(np.mean(cohen_ece_results_dict["cohen_kappa"][model_name]))
                    evolution_dict["ece"][model_name][dataset][real_sample_size].append(
                        np.mean(cohen_ece_results_dict["ece"][model_name])
                    )
                    evolution_dict[model_name][dataset][real_sample_size].append(
                        np.mean(train_test_times_cur[model_name])
                    )
                    
                    if (
                        len(
                            evolution_dict["cohen_kappa"][model_name][dataset][
                                real_sample_size
                            ]
                        )
                        == reps
                    ):  # Changing the results to tuple enabling easier saving to txt / json and ectacting the fata after that.
                        
                            
                        train_test_times[model_name][dataset][real_sample_size] = tuple(train_test_times[model_name][dataset][real_sample_size])
                            
                        evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size] = 
                        tuple(evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size]  )
                        
                        evolution_dict["ece"][model_name][dataset][
                            real_sample_size
                        ] = tuple(
                            evolution_dict["ece"][model_name][dataset][real_sample_size]
                        )
                    # else:
                    #    del evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size]
        k_index += 1

    # Record sample sizes used
    all_sample_sizes[dataset_index][:] = np.array(training_sample_sizes)

new_dict = {}
for key_met in evoluton_dict.keys():
    new_dict[key_met] = mod_dict(evolution_dict[key_met], tuple)
new_dict_times = mod_dict(train_test_times, tuple)    


# Save sample sizes and model results in txt files
np.savetxt("metrics/cc18_sample_sizes.txt", all_sample_sizes)
save_methods = {"text_dict": 1, "csv": 0, "json": 0}
save_methods_rewrite = {"text_dict": 1, "csv": 0, "json": 0}

save_best_parameters(
    save_methods, save_methods_rewrite, "results/cc18_kappa_and_ece", new_dict
)
save_best_parameters(
    save_methods, save_methods_rewrite, "results/cc18_training_times", new_dict_times
)

