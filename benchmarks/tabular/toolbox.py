"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Haoyin Xu
"""
import numpy as np
from random import sample
import ast
import openml


def load_cc18():
    """
    Import datasets from OpenML-CC18 dataset suite
    """
    X_data_list = []
    y_data_list = []
    dataset_name = []

    for data_id in openml.study.get_suite("OpenML-CC18").data:
        try:
            successfully_loaded = True
            dataset = openml.datasets.get_dataset(data_id, download_data=False)
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


def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    """
    Return ece score

    Function borrowed from: https://github.com/neurodata/kdg/blob/main/kdg/utils.py
    """
    poba_hist = []
    accuracy_hist = []
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


def sample_large_datasets(X_data, y_data):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples.
    """
    inds = [i for i in range(X_data.shape[0])]
    fin = sorted(sample(inds, 10000))
    return X_data[fin], y_data[fin]
