"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Haoyin Xu
"""
from toolbox import *

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


# Import CC18 data and pretuned hyperparameters
X_data_list, y_data_list, dataset_name = load_cc18()
all_params = read_params_txt("metrics/cc18_all_parameters.txt")
train_indices = [i for i in range(72)]


# Number of repetitions for each CV fold at each sample size
reps = 5


# Create empty arrays to index sample sizes, kappa values, and ece scores
all_sample_sizes = np.zeros((len(train_indices), 8))

rf_evolution = np.zeros((8 * len(train_indices), 5))
dn_evolution = np.zeros((8 * len(train_indices), 5))

rf_evolution_ece = np.zeros((8 * len(train_indices), 5))
dn_evolution_ece = np.zeros((8 * len(train_indices), 5))


# For each dataset, train and predict for every sample size
# Record outputs using Cohen's Kappa and ECE
for dataset_index, dataset in enumerate(train_indices):

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

            rf_reps = np.zeros((reps))
            dn_reps = np.zeros((reps))

            rf_reps_ece = np.zeros((reps))
            dn_reps_ece = np.zeros((reps))

            # Repeat for number of repetitions, averaging results
            for ii in range(reps):
                rf = RandomForestClassifier(
                    **all_params[dataset][1], n_estimators=500, n_jobs=-1
                )
                mlp = MLPClassifier(**all_params[dataset][0])

                rf.fit(X_train_new, y_train_new)
                y_pred_rf = rf.predict(X_test)
                proba_rf = rf.predict_proba(X_test)

                mlp.fit(X_train_new, y_train_new)
                y_pred = mlp.predict(X_test)
                proba_dn = mlp.predict_proba(X_test)

                k_rf = cohen_kappa_score(y_test, y_pred_rf)
                rf_reps[ii] = k_rf

                k = cohen_kappa_score(y_test, y_pred)
                dn_reps[ii] = k

                ece_rf = get_ece(proba_rf, y_pred_rf, y_test)
                rf_reps_ece[ii] = ece_rf

                ece_dn = get_ece(proba_dn, y_pred, y_test)
                dn_reps_ece[ii] = ece_dn

            # Record Cohen's Kappa score and ECE for both RF and DN
            rf_evolution[sample_size_index + 8 * dataset_index][k_index] = np.mean(
                rf_reps
            )
            dn_evolution[sample_size_index + 8 * dataset_index][k_index] = np.mean(
                dn_reps
            )
            rf_evolution_ece[sample_size_index + 8 * dataset_index][k_index] = np.mean(
                rf_reps_ece
            )
            dn_evolution_ece[sample_size_index + 8 * dataset_index][k_index] = np.mean(
                dn_reps_ece
            )

        k_index += 1

    # Record sample sizes used
    all_sample_sizes[dataset_index][:] = np.array(training_sample_sizes)


# Save sample sizes and model results in txt files
np.savetxt("metrics/cc18_sample_sizes.txt", all_sample_sizes)
np.savetxt("results/cc18_dn_kappa.txt", dn_evolution)
np.savetxt("results/cc18_rf_kappa.txt", rf_evolution)
np.savetxt("results/cc18_dn_ece.txt", dn_evolution_ece)
np.savetxt("results/cc18_rf_ece.txt", rf_evolution_ece)
