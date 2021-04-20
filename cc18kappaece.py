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
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import cohen_kappa_score
import ast

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
     
    
sam = []
feat = []
for i in X_data_list:
    sam.append(i.shape[0])
    feat.append(i.shape[1])
    print('Samples: ', i.shape[0])
    print('Features: ', i.shape[1])
    print('\n')

    
unique_classes = []
for i in y_data_list:
    unique_classes.append(len(np.unique(i)))
print(unique_classes)

count = 0
for i in unique_classes:
    if i == 2:
        count += 1
print(count)



def load_result(filename):
    """
    Loads results from specified file
    """
    return np.loadtxt(filename)

def read_params_txt(filename):
    params = []
    f = open(filename, 'r').read()
    f = f.split('\n')
    f = f[:-1]
    for ind, i in enumerate(f):
        temp = ast.literal_eval(f[ind])
        params.append(temp)
    return params

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

def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    poba_hist = []
    accuracy_hist = []
    bin_size = 1/num_bins
    total_sample = len(true_label)
    posteriors = predicted_posterior.max(axis=1)

    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (posteriors>bin*bin_size) & (posteriors<=(bin+1)*bin_size)
        )[0]
        
        acc = np.nan_to_num(
            np.mean(
            predicted_label[indx] == true_label[indx]
        )
        ) if indx.size!=0 else 0
        conf = np.nan_to_num(
            np.mean(
            posteriors[indx]
        )
        ) if indx.size!=0 else 0
        score += len(indx)*np.abs(
            acc - conf
        )
    
    score /= total_sample
    return score



all_params = read_params_txt("all_parameters_final.txt")


small_datasets = [i for i,value in enumerate(sam) if value < 10000]



all_sample_sizes = np.zeros((len(small_datasets), 8))

reps = 5

rf_evolution = np.zeros((8*len(small_datasets),5))
dn_evolution = np.zeros((8*len(small_datasets),5))

rf_evolution_ece = np.zeros((8*len(small_datasets),5))
dn_evolution_ece = np.zeros((8*len(small_datasets),5))

for dataset_index, dataset in enumerate(small_datasets):
    
    print('\n\n\n\nDATASET: ', dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
        
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    k_index=0
    for train_index, test_index in kf.split(X,y):
        print('CV Fold: ', k_index)
        print('\n')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  
    
        temp = np.log10((len(np.unique(y))) * 5)
        t = (np.log10(X_train.shape[0]) - temp) / 7
        training_sample_sizes = []
        for i in range(8):
            training_sample_sizes.append(round(np.power(10,temp + i*t)))
    
        ss_inds = random_sample_new(X_train, training_sample_sizes)
        
        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):
            print('Sample Size Index: ', sample_size_index)
            
            X_train_new = X_train[ss_inds[sample_size_index]]
            y_train_new = y_train[ss_inds[sample_size_index]]

            rf_reps = np.zeros((reps))
            dn_reps = np.zeros((reps))

            rf_reps_ece = np.zeros((reps))
            dn_reps_ece = np.zeros((reps))
            
            for ii in range(reps):
#                 print('Repetition: ', ii)
                rf = RandomForestClassifier(**all_params[dataset][1], n_estimators=500)
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
            
            rf_evolution[sample_size_index + 8*dataset_index][k_index] = np.mean(rf_reps)
            dn_evolution[sample_size_index + 8*dataset_index][k_index] = np.mean(dn_reps)
            rf_evolution_ece[sample_size_index + 8*dataset_index][k_index] = np.mean(rf_reps_ece)
            dn_evolution_ece[sample_size_index + 8*dataset_index][k_index] = np.mean(dn_reps_ece)
            
        k_index += 1
    
    all_sample_sizes[dataset_index][:] = np.array(training_sample_sizes)


print(dn_evolution)
print(rf_evolution)

print(dn_evolution_ece)
print(rf_evolution_ece)



np.savetxt('sample_sizes_ecekappa_small.txt', all_sample_sizes)     

np.savetxt('dn_evolution_kappa_small.txt', dn_evolution)     
np.savetxt('rf_evolution_kappa_small.txt', rf_evolution)   
np.savetxt('dn_evolution_ece_small.txt', dn_evolution_ece)     
np.savetxt('rf_evolution_ece_small.txt', rf_evolution_ece)   
