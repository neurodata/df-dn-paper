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

from basic_functions_script import *
from save_hyperparameters import *

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
shape_2_all_sample_sizes = 8
shape_2_evolution = 5
n_splits = 5

models_to_run = {"RF": 1, "DN": 1, "GBDT": 1}


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


def sample_large_datasets(X_data, y_data, max_shape_to_run=10000):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, max_shape_to_run))
    return X_data[fin], y_data[fin]


# Import CC18 data and pretuned hyperparameters
if reload_data or "dataset_name" not in locals():
    X_data_list, y_data_list, dataset_name = load_cc18()

best_params_dict = read_params_dict_txt(path, file_type_to_load)


# all_params = read_params_txt("metrics/cc18_all_parameters.txt")

train_indices = [i for i in range(dataset_indices_max)]


# Create empty arrays to index sample sizes, kappa values, and ece scores
all_sample_sizes = np.zeros((len(train_indices), shape_2_all_sample_sizes))
evolution_dict = {
    metric: {
        model_name: {}
        for model_name in best_params_dict.keys()
    }
    for metric in ["cohen_kappa", "ece"]
}


# For each dataset, train and predict for every sample size
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
                    metric: {model_name: np.zeros((reps)) for model_name in best_params_dict.keys()}
                    for metric in ["cohen_kappa", "ece"]}
            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            # Repeat for number of repetitions, averaging results
            for model_name, model_best_params in best_params_dict.items():
                if models_to_run[model_name]:
                    for ii in range(reps):

                        model = model_define(model_name, best_params_dict, dataset_index )
                        model.fit(X_train_new, y_train_new)
                        predictions = model.predict(X_test)
                        predict_probas = model.predict_proba(X_test)
                        
                        cohen_kappa = cohen_kappa_score(y_test, predictions)
                        cohen_ece_results_dict["cohen_kappa"][model_name][
                            ii
                        ] = cohen_kappa
                        ece = get_ece(predict_probas, predictions, y_test)
                        cohen_ece_results_dict["ece"][model_name][ii] = ece
                    if dataset not in evolution_dict["cohen_kappa"][model_name].keys():
                        evolution_dict["cohen_kappa"][model_name][dataset] = {}
                        evolution_dict["ece"][model_name][dataset] = {}
                    if real_sample_size not in evolution_dict["cohen_kappa"][model_name][dataset].keys():
                        evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size] = []
                        evolution_dict["ece"][model_name][dataset][real_sample_size] = []
                        
                    evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size].append(np.mean(
                        cohen_ece_results_dict["cohen_kappa"][model_name]
                    ))
                    evolution_dict["ece"][model_name][dataset][real_sample_size].append(np.mean(
                        cohen_ece_results_dict["ece"][model_name]
                    ))
                    if len(evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size])==reps: # Changing the results to tuple enabling easier saving to txt / json and ectacting the fata after that. 
                        evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size] = tuple(evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size])
                        evolution_dict["ece"][model_name][dataset][real_sample_size] = tuple(evolution_dict["ece"][model_name][dataset][real_sample_size])
                    else: 
                        del evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size]
        k_index += 1

    # Record sample sizes used
    all_sample_sizes[dataset_index][:] = np.array(training_sample_sizes)

new_dict = {}
for key_met in evoluton_dict.keys():
    new_dict[key_met] = mod_dict(evolution_dict[key_met],tuple)




# Save sample sizes and model results in txt files
np.savetxt("metrics/cc18_sample_sizes.txt", all_sample_sizes)
save_methods = {"text_dict": 1, "csv": 0, "json": 0}
save_methods_rewrite = {"text_dict": 1, "csv": 0, "json": 0}


#dictionary_test = {key:{key2:tuple(value) for key2,value in inner_res.items()} for key,inner_res in evolution_dict.items()}

save_best_parameters(
    save_methods, save_methods_rewrite, "metrics/kappa_and_ece", new_dict
)
