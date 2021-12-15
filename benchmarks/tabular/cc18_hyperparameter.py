"""
Coauthors: Michael Ainsworth
           Haoyin Xu
           Noga Mudrik
"""
# Imports

from sklearn.preprocessing import StandardScaler
import itertools
from toolbox import *


# Define executrion variables (to save memory & execution time)
reload_data = False  # indicator of whether to upload the data again
return_default = False

nodes_combination = [20, 100, 180, 260, 340, 400]
dataset_indices_max = 72
max_shape_to_run = 10000
alpha_range_nn = [0.1, 0.001, 0.01, 0.1]
subsample = [1.0, 0.5, 0.8, 1.0]
path_save = "metrics/cc18_all_parameters"
path_save_dict_data_indices = "metrics/dict_data_indices"
save_methods = {"text_dict": 0, "csv": 0, "json": 1}
save_methods_rewrite = {"text_dict": 0, "csv": 0, "json": 1}

if return_default:
    (
        nodes_combination,
        dataset_indices_max,
        max_shape_to_run,
        models_to_run,
        subsample,
        alpha_range_nn,
    ) = return_to_default()

"""
Models:
Deep Neural Network
"""
# Generate all combinations of nodes to tune over
test_list = nodes_combination
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer
"""
Change below to add a model
"""
models_to_run = {
    "RF": 1,
    "DN": 1,
    "GBDT": 1,
}  # Define which models to run

classifiers = {
    "DN": TabNetClassifier(),
    "RF": RandomForestClassifier(n_estimators=500),
    "GBDT": xgb.XGBClassifier(booster="gbtree", base_score=0.5),
}


varargin = {
    "node_range": node_range,
    "alpha_range_nn": alpha_range_nn,
    "subsample": subsample,
}

varCV = {
    "DN": {"n_jobs": -1, "verbose": 1},
    "RF": {"n_jobs": -1, "verbose": 1},
    "GBDT": {"n_jobs": -1, "verbose": 1},
}

save_vars_to_dict(
    classifiers,
    varargin,
    reload_data,
    nodes_combination,
    dataset_indices_max,
    max_shape_to_run,
    alpha_range_nn,
    subsample,
    "metrics/dict_parameters.json",
)


"""
function to save cc18_all_parameters file
Empty dict to record optimal parameters
"""

all_parameters = {
    model_name: {} for model_name, val in models_to_run.items() if val == 1
}
best_parameters = {
    model_name: {} for model_name, val in models_to_run.items() if val == 1
}
all_params = {model_name: {} for model_name, val in models_to_run.items() if val == 1}

"""
Organize the data
"""
# Load data from CC18 data set suite
if (
    reload_data or "dataset_name" not in locals()
):  # Load the data only if required (by reload_data or if it is not defined)
    X_data_list, y_data_list, dataset_name = load_cc18()

"""
Choose dataset indices
"""

dataset_indices = list(range(dataset_indices_max))
dict_data_indices = {dataset_ind: {} for dataset_ind in dataset_indices}


"""
For each dataset, use randomized hyperparameter search to optimize parameters
"""

for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    """
    If data set has over 10000 samples, resample to contain 10000
    """

    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)

    """
    Split to train, val and test by ratio of 2:1:1
    """
    np.random.seed(dataset_index)
    dict_data_indices = find_indices_train_val_test(
        X.shape[0], dict_data_indices=dict_data_indices, dataset_ind=dataset_index
    )
    train_indices = dict_data_indices[dataset_index]["train"]
    val_indices = dict_data_indices[dataset_index]["val"]
    """
    Standart Scaling
    """
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    p = X.shape[1]

    for model_name, val_run in models_to_run.items():
        if val_run == 1:
            if model_name not in classifiers:
                raise ValueError(
                    "Model name is not defined in the classifiers dictionary"
                )
            else:
                all_parameters, best_parameters, all_params = do_calcs_per_model(
                    all_parameters,
                    best_parameters,
                    all_params,
                    model_name,
                    varargin,
                    classifiers,
                    X,
                    y,
                    dataset_index,
                    train_indices,
                    val_indices,
                    p,
                    varCV,
                )

save_best_parameters(save_methods, save_methods_rewrite, path_save, best_parameters)
save_best_parameters(
    save_methods, save_methods_rewrite, path_save_dict_data_indices, dict_data_indices
)
