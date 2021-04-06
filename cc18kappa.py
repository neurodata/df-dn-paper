import numpy as np
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 6

import seaborn as sns
import pandas as pd
from random import choices
from random import sample

from tqdm.notebook import tqdm

from scipy import stats

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import cohen_kappa_score

import openml



X_data_list = []
y_data_list = []
dataset_name = []

for task_num, task_id in enumerate(tqdm(openml.study.get_suite("OpenML-CC18").tasks)):
    try:
        successfully_loaded = True
        dataset = openml.datasets.get_dataset(openml.tasks.get_task(task_id).dataset_id)
        print(dataset)
        dataset_name.append(dataset.name)
        X, y, is_categorical, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
        _, y = np.unique(y, return_inverse = True)
        #X = np.nan_to_num(X[:, np.where(np.array(is_categorical) == False)[0]])
        X = np.nan_to_num(X)
    except TypeError:
        print("Skipping Dataset {}".format(dataset_idx))
        print()
        successfully_loaded = False
    if successfully_loaded and np.shape(X)[1] > 0:
        print('\n\nSuccess: ', task_num)
        X_data_list.append(X)
        y_data_list.append(y)




def random_sample_new(data, training_sample_sizes):
    
    temp_inds = []

    ordered = [i for i in range(len(data))]
    minus = 0
    for ss in range(len(training_sample_sizes)):
        x = sorted(sample(ordered,training_sample_sizes[ss] - minus))
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


import itertools

test_list = [20,100,180,260,340,400]
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer



all_parameters = []

num_datasets = 3

all_sample_sizes = np.zeros((num_datasets, 8))

rf_evolution = np.zeros((8*num_datasets,5))
dn_evolution = np.zeros((8*num_datasets,5))

for dataset in range(num_datasets):
    
    print('Dataset: ', dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    parameters = {
        'hidden_layer_sizes' : node_range,
        'alpha' : [0.0001,0.001,0.01,0.1]
    }

    p = X.shape[1]
    l = list(set([round(p/4),round(np.sqrt(p)),round(p/3),round(p/1.5),round(p)]))
    parameters_rf = {
        'max_features' : l
    }

    mlp = MLPClassifier(max_iter=200)
    clf = RandomizedSearchCV(mlp, parameters, n_jobs=-1, cv=None, verbose=1)
    clf.fit(X, y)

    rf = RandomForestClassifier(n_estimators=500)
    clfrf = RandomizedSearchCV(rf, parameters_rf, n_jobs=-1, verbose=1)
    clfrf.fit(X, y)

    allparams = clf.cv_results_['params']
    allparamsrf = clfrf.cv_results_['params']

    best_params = clf.best_params_
    best_paramsrf = clfrf.best_params_

    all_parameters.append([best_params, best_paramsrf])

    
    kf = KFold(n_splits=5, shuffle=True)

    k_index=0
    for train_index, test_index in kf.split(X):
        print('CV Fold: ', k_index)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  
    
        temp = np.log10((len(np.unique(y))) * 5)
        t = (np.log10(X_train.shape[0]) - temp) / 7
        training_sample_sizes = []
        for i in range(8):
            training_sample_sizes[dataset][i] = round(np.power(10,temp + i*t))
        print(training_sample_sizes)
    
        ss_inds = random_sample_new(X_train, training_sample_sizes)


        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):
            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            rf = RandomForestClassifier(**best_paramsrf, n_estimators=500)
            mlp = MLPClassifier(**best_params)

            rf.fit(X_train_new, y_train_new)
            y_pred_rf = rf.predict(X_test)

            k_rf = cohen_kappa_score(y_test, y_pred_rf)
            rf_evolution[sample_size_index + 8*dataset][k_index] = k_rf


            mlp.fit(X_train_new, y_train_new)
            y_pred = mlp.predict(X_test)

            k = cohen_kappa_score(y_test, y_pred)
            dn_evolution[sample_size_index + 8*dataset][k_index] = k
            
        k_index += 1
    
    all_sample_sizes.append(training_sample_sizes)

print(dn_evolution)
print(rf_evolution)
    

np.savetxt('sample_sizes.txt', all_sample_sizes)     

np.savetxt('dn_evolution.txt', dn_evolution)     
np.savetxt('rf_evolution.txt', rf_evolution)   

with open('all_parameters.txt', 'w') as f:
    for item in all_parameters:
        f.write("%s\n" % item)