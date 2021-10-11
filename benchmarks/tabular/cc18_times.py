"""
Coauthors: Michael Ainsworth
           Haoyin Xu
"""
from toolbox import *

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import time


# Load data from CC18 data set suite
X_data_list, y_data_list, dataset_name = load_cc18()
dataset_indices = [i for i in range(72)]

# Import pretuned hyperparameters
all_params = read_params_txt("metrics/cc18_all_parameters.txt")


# Empty arrays to index times into
rf_times_train = np.zeros((8 * len(dataset_indices), 5))
rf_times_test = np.zeros((8 * len(dataset_indices), 5))

dn_times_train = np.zeros((8 * len(dataset_indices), 5))
dn_times_test = np.zeros((8 * len(dataset_indices), 5))


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

            rf = RandomForestClassifier(
                **all_params[dataset][1], n_estimators=500, n_jobs=-1
            )

            start_time = time.perf_counter()
            rf.fit(X_train_new, y_train_new)
            end_time = time.perf_counter()
            rf_train_time = end_time - start_time

            start_time = time.perf_counter()
            y_pred_rf = rf.predict(X_test)
            end_time = time.perf_counter()
            rf_test_time = end_time - start_time

            mlp = MLPClassifier(**all_params[dataset][0])

            start_time = time.perf_counter()
            mlp.fit(X_train_new, y_train_new)
            end_time = time.perf_counter()
            dn_train_time = end_time - start_time

            start_time = time.perf_counter()
            y_pred_mlp = mlp.predict(X_test)
            end_time = time.perf_counter()
            dn_test_time = end_time - start_time

            rf_times_train[sample_size_index + 8 * dataset_index][
                k_index
            ] = rf_train_time
            rf_times_test[sample_size_index + 8 * dataset_index][k_index] = rf_test_time

            dn_times_train[sample_size_index + 8 * dataset_index][
                k_index
            ] = dn_train_time
            dn_times_test[sample_size_index + 8 * dataset_index][k_index] = dn_test_time

        k_index += 1


# Save results as txt files
np.savetxt("results/cc18_rf_times_train.txt", rf_times_train)
np.savetxt("results/cc18_rf_times_test.txt", rf_times_test)
np.savetxt("results/cc18_dn_times_train.txt", dn_times_train)
np.savetxt("results/cc18_dn_times_test.txt", dn_times_test)
