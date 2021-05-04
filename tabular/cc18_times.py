"""
Author: Michael Ainsworth
"""

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import ast
import openml
import timeit


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


def time_rf_cc18(SETUP_CODE, X_train_new, y_train_new, X_test, all_params, dataset):
    """
    Use timeit function to measure training and fitting time of the Random Forest
    classifier using optimized parameters for given dataset.
    """
    TEST_CODE = """

rf = RandomForestClassifier(**{}[{}][1], n_estimators=500, n_jobs=-1)
rf.fit(X_train_new, y_train_new)
y_pred_rf = rf.predict(X_test)

    """.format(
        all_params, dataset
    )
    time_rf = timeit.repeat(
        setup=SETUP_CODE_RF,
        stmt=TEST_CODE,
        repeat=1,
        number=1,
        globals={
            "X_train_new": X_train_new,
            "y_train_new": y_train_new,
            "X_test": X_test,
        },
    )

    return time_rf


SETUP_CODE_RF = """

from sklearn.ensemble import RandomForestClassifier

"""


def time_mlp_cc18(SETUP_CODE, X_train_new, y_train_new, X_test, all_params, dataset):
    """
    Use timeit function to measure training and fitting time of the Random Forest
    classifier using optimized parameters for given dataset.
    """
    TEST_CODE = """

mlp = MLPClassifier(**{}[{}][0])
mlp.fit(X_train_new, y_train_new)
y_pred_mlp = mlp.predict(X_test)

    """.format(
        all_params, dataset
    )
    time_mlp = timeit.repeat(
        setup=SETUP_CODE_MLP,
        stmt=TEST_CODE,
        repeat=1,
        number=1,
        globals={
            "X_train_new": X_train_new,
            "y_train_new": y_train_new,
            "X_test": X_test,
        },
    )

    return time_mlp


SETUP_CODE_MLP = """

from sklearn.neural_network import MLPClassifier

"""


# Load data from CC18 data set suite
X_data_list, y_data_list, dataset_name = load_cc18()
# dataset_indices = [i for i in range(72)]
dataset_indices = [0, 2]

# Import pretuned hyperparameters
all_params = read_params_txt("metrics/cc18_all_parameters.txt")


# Empty arrays to index times into
rf_times = np.zeros((8 * len(dataset_indices), 5))
dn_times = np.zeros((8 * len(dataset_indices), 5))


# For each dataset, determine wall times at each sample size
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
        for i in range(8):
            training_sample_sizes.append(round(np.power(10, temp + i * t)))

        ss_inds = random_sample_new(X_train, training_sample_sizes)

        # Iterate through each sample size per dataset
        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):

            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            rf_time_temp = time_rf_cc18(
                SETUP_CODE_RF, X_train_new, y_train_new, X_test, all_params, dataset
            )
            mlp_time_temp = time_mlp_cc18(
                SETUP_CODE_MLP, X_train_new, y_train_new, X_test, all_params, dataset
            )

            rf_times[sample_size_index + 8 * dataset_index][k_index] = rf_time_temp[0]
            dn_times[sample_size_index + 8 * dataset_index][k_index] = mlp_time_temp[0]

        k_index += 1


# Save results as txt files
np.savetxt("results/cc18_rf_times.txt", rf_times)
np.savetxt("results/cc18_dn_times.txt", dn_times)
