"""
Coauthors: Haoyin Xu
"""
# Imports
import argparse
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# from xgboost import XGBClassifier

from toolbox import *

MAX_SAMPLES = 10000

# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-rf", help="random forests", required=False, action="store_true")
parser.add_argument(
    "-gb", help="gradient boosting trees", required=False, action="store_true"
)
parser.add_argument("-dn", help="deep networks", required=False, action="store_true")
args = parser.parse_args()

# Collect dataset IDs
datasets = openml.study.get_suite("OpenML-CC18").data

# Create 5 dataset meta-folds
metaKF = KFold(shuffle=True)

if args.all or args.rf:
    # random forests
    rf_kappa_l = []
    rf_ece_l = []
    rf_tune_time_l = []
    rf_train_time_l = []
    rf_test_time_l = []

# For each meta-fold
for train_index, test_index in metaKF.split(datasets):
    train_datasets = np.array(datasets)[train_index]
    test_datasets = np.array(datasets)[test_index]

    if args.all or args.rf:
        best_acc = 0
        best_param_combo = {}
        acc_ls = []

        # Record tuning time start
        start_tune_time = time.perf_counter()

        # For each parameter combo
        param_combo_l = param_list(rf_params())
        for param_combo in param_combo_l:
            acc_l = []

            # For each training dataset
            for dataset in train_datasets:
                # Load dataset
                X, y = load_dataset(dataset, MAX_SAMPLES)

                # Conduct 5-fold cross validations
                scores = cross_validate(
                    RandomForestClassifier(param_combo), X, y, scoring="accuracy"
                )
                acc_l.append(scores["test_score"])

            acc_ls.append(acc_l)

        # Select best performing parameters
        for i in range(len(param_combo_l)):
            acc = np.mean(acc_ls[i])
            if acc >= best_acc:
                best_acc = acc
                best_param_combo = param_combo_l[i]

        # Record tuning time end
        end_tune_time = time.perf_counter()
        tune_time = end_tune_time - start_tune_time
        rf_tune_time_l.append(tune_time)

        # TODO: test best parameters performance on testing meta-fold datasets
        clf = RandomForestClassifier(best_param_combo)

        # Record training & testing time start
        start_train_time = 0
        start_test_time = 0

        # For each testing dataset
        for dataset in test_datasets:
            # Load dataset
            X, y = load_dataset(dataset, MAX_SAMPLES)
