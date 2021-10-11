"""
Coauthors: Michael Ainsworth
           Haoyin Xu
"""
from toolbox import *

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import itertools


# Load data from CC18 data set suite
X_data_list, y_data_list, dataset_name = load_cc18()


# Generate all combinations of nodes to tune over
test_list = [20, 100, 180, 260, 340, 400]
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer


# Empty list to record optimal parameters
all_parameters = []


# Choose dataset indices
dataset_indices = [i for i in range(72)]


# For each dataset, use randomized hyperparameter search to optimize parameters
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > 10000:
        X, y = sample_large_datasets(X, y)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Deep network hyperparameters
    parameters = {"hidden_layer_sizes": node_range, "alpha": [0.0001, 0.001, 0.01, 0.1]}

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
