"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Haoyin Xu
           Noga Mudrik
           Audrey Zheng
"""
import numpy as np
import openml
import json
from os.path import exists
from sklearn.model_selection import (
    KFold,
    cross_validate,
    ParameterGrid,
    # ParameterSampler,
    # RandomizedSearchCV,
    train_test_split,
)


def sample_large_datasets(X_data, y_data, max_size=10000):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    X_data, _, y_data, _ = train_test_split(
        X_data, y_data, train_size=max_size, stratify=y_data
    )
    return X_data, y_data


def param_list(param_dict):
    """Produce parameter grids"""
    return list(ParameterGrid(param_dict))


def rf_params():
    """RF Hyperparameters for tuning"""
    n_estimators = np.linspace(50, 500, num=10)
    criterion = ["gini", "entropy"]
    max_features = ["sqrt", "log2", None]
    min_samples_split = [2, 6, 10]
    max_samples = [0.5, None]
    param_dict = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "min_samples_split": min_samples_split,
        "max_samples": max_samples,
        "n_jobs": [-1],
    }

    return param_dict


# def xgb_params():
#     booster = "gbtree"
#     n_estimators = np.linspace(50, 500, num=10)
#     learning_rate = [0.05, 0.1, 0.3, 0.5]
#
#     min_samples_split = [2, 6, 10]
#     max_samples = [0.5, None]
#     param_dict = {
#         "booster": booster,
#         "n_estimators": n_estimators,
#         "criterion": criterion,
#         "min_samples_split": min_samples_split,
#         "max_samples": max_samples,
#         "n_jobs": [-1]
#     }
#
#     return param_dict
#
# def save_best_parameters(
#     save_methods, save_methods_rewrite, path_save, best_parameters
# ):
#     """
#     Save Hyperparameters
#     """
#     if exists(path_save + ".json") and save_methods_rewrite["json"] == 0:
#         with open(path_save + ".json", "r") as json_file:
#             dictionary = json.load(json_file)
#         best_parameters_to_save = {**dictionary, **best_parameters}
#     else:
#         best_parameters_to_save = best_parameters
#     with open(path_save + ".json", "w") as fp:
#         json.dump(best_parameters_to_save, fp)
#
#
#
# def do_calcs_per_model(
#     all_parameters,
#     best_parameters,
#     all_params,
#     model_name,
#     varargin,
#     classifiers,
#     X,
#     y,
#     dataset_index,
#     train_indices,
#     val_indices,
#     p=None,
# ):
#     """
#     find best parameters for given sample_size, dataset_index, model_name
#     """
#     model = classifiers[model_name]
#     parameters = create_parameters(model_name, varargin, p)
#     clf = RandomizedSearchCV(model, parameters, n_jobs=-1)
#     clf.fit(X, y)
#     all_parameters[model_name][dataset_index] = parameters
#     best_parameters[model_name][dataset_index] = clf.best_params_
#     all_params[model_name][dataset_index] = clf.cv_results_["params"]
#     return all_parameters, best_parameters, all_params
#
#
# def load_cc18():
#     """
#     Import datasets from OpenML-CC18 dataset suite
#     """
#     X_data_list = []
#     y_data_list = []
#     dataset_name = []
#
#     for data_id in openml.study.get_suite("OpenML-CC18").data:
#         try:
#             successfully_loaded = True
#             dataset = openml.datasets.get_dataset(data_id)
#             dataset_name.append(dataset.name)
#             X, y, is_categorical, _ = dataset.get_data(
#                 dataset_format="array", target=dataset.default_target_attribute
#             )
#             _, y = np.unique(y, return_inverse=True)
#             X = np.nan_to_num(X)
#         except TypeError:
#             successfully_loaded = False
#         if successfully_loaded and np.shape(X)[1] > 0:
#             X_data_list.append(X)
#             y_data_list.append(y)
#
#     return X_data_list, y_data_list, dataset_name
#
#
# def return_default():
#     """
#     function to return to default values
#     """
#     nodes_combination = [20, 100, 180, 260, 340, 400]
#     dataset_indices_max = 72
#     alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
#     subsample = [0.5, 0.8, 1.0]
#     return (
#         nodes_combination,
#         dataset_indices_max,
#         models_to_run,
#         subsample,
#         alpha_range_nn,
#     )
#
#
# def save_vars_to_dict(
#     classifiers,
#     varargin,
#     reload_data=False,
#     nodes_combination=[20],
#     dataset_indices_max=2,
#     max_shape_to_run=10000,
#     alpha_range_nn=[0.1],
#     subsample=[1.0],
#     path_to_save="metrics/dict_parameters.json",
#     shape_2_evolution=1,
#     shape_2_all_sample_sizes=8,
# ):
#     """
#     save variables to dict
#     Function to save variables to dict for later use
#     """
#     dict_to_save = {
#         "reload_data": reload_data,
#         "nodes_combination": nodes_combination,
#         "dataset_indices_max": dataset_indices_max,
#         "max_shape_to_run": max_shape_to_run,
#         "alpha_range_nn": alpha_range_nn,
#         "subsample": subsample,
#         "classifiers_names": tuple(classifiers.keys()),
#         "varargin": varargin,
#         "shape_2_evolution": shape_2_evolution,
#         "shape_2_all_sample_sizes": shape_2_all_sample_sizes,
#     }
#
#     with open(path_to_save, "w") as fp:
#         json.dump(dict_to_save, fp)
#
#
# def mod_dict(res_dict, type_to_compare):
#     tot_dict = {}
#     for model_name in res_dict.keys():
#         modified_subdict = {}
#         for dataset in res_dict[model_name].keys():
#             data_set_dict = res_dict[model_name][dataset]
#             data_set_dict = {
#                 key: val
#                 for key, val in data_set_dict.items()
#                 if isinstance(val, type_to_compare)
#             }
#             modified_subdict[dataset] = data_set_dict
#         tot_dict[model_name] = modified_subdict
#     return tot_dict
#
#
# def read_params_dict_json(path="metrics/cc18_all_parameters", type_file=".json"):
#     """
#     Read optimized parameters as saved in a dict
#     Path should not include the file type (like .json)
#     """
#
#     with open(path + ".json", "r") as json_file:
#         best_params_dict = json.load(json_file)
#
#     return best_params_dict
#
#
# def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
#     """
#     Return ece score
#     Function borrowed from: https://github.com/neurodata/kdg/blob/main/kdg/utils.py
#     """
#
#     bin_size = 1 / num_bins
#     total_sample = len(true_label)
#     posteriors = predicted_posterior.max(axis=1)
#
#     score = 0
#     for bin in range(num_bins):
#         indx = np.where(
#             (posteriors > bin * bin_size) & (posteriors <= (bin + 1) * bin_size)
#         )[0]
#
#         acc = (
#             np.nan_to_num(np.mean(predicted_label[indx] == true_label[indx]))
#             if indx.size != 0
#             else 0
#         )
#         conf = np.nan_to_num(np.mean(posteriors[indx])) if indx.size != 0 else 0
#         score += len(indx) * np.abs(acc - conf)
#
#     score /= total_sample
#     return score
#
#
# def find_indices_train_val_test(
#     X_shape,
#     ratio=[2, 1, 1],
#     keys_types=["train", "val", "test"],
#     dict_data_indices={},
#     dataset_ind=0,
# ):
#     ratio_base = [int(el) for el in np.linspace(0, X_shape, np.sum(ratio) + 1)]
#     ratio_base_limits = np.hstack([[0], ratio_base[np.cumsum(ratio)]])
#     list_indices = np.arange(X_shape)
#     np.random.shuffle(list_indices)
#     for ind_counter, ind_min in enumerate(ratio_base_limits[:-1]):
#         ind_max = ratio_base_limits[ind_counter + 1]
#         cur_indices = list_indices[ind_min:ind_max]
#         dict_data_indices[dataset_ind][keys_types[ind_counter]] = cur_indices
#     return dict_data_indices
