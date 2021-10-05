"""
Author: Michael Ainsworth
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm.notebook import tqdm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import openml
import itertools
import pandas as pd
import json
from tqdm import tqdm
from os.path import exists
import ast
import os

from basic_functions_script import *
from save_hyperparameterst import *
#%% function to return to default values
def return_to_default():
    nodes_combination = [20, 100, 180, 260, 340, 400]
    dataset_indices_max = 72 
    max_shape_to_run = 10000
    alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
    subsample=[0.5,0.8,1.0]
    models_to_run={'RF':1,'DN':1,'GBDT':1}
    return nodes_combination,dataset_indices_max,max_shape_to_run,models_to_run,subsample,alpha_range_nn

#%% Define executrion variables (to save memory & execution time)
reload_data = False # indicator of whether to upload the data again
nodes_combination = [20]#[20, 100, 180, 260, 340, 400] # default is [20, 100, 180, 260, 340, 400]
dataset_indices_max=2 #72 
max_shape_to_run=10000
alpha_range_nn=[0.1] #, 0.001, 0.01, 0.1]
subsample=[1.0] #[0.5,0.8,1.0]

return_default=False;
if return_default:
    nodes_combination,dataset_indices_max,max_shape_to_run,models_to_run,subsample,alpha_range_nn=return_to_default()
path_save="metrics/cc18_all_parameters"

save_methods={'text_dict':1,'csv':1,'json':1}
save_methods_rewrite={'text_dict':0,'csv':0,'json':0}
#%% Models
"""
Deep Neural Network
"""
# Generate all combinations of nodes to tune over
test_list = nodes_combination;
two_layer = list(itertools.combinations(test_list, 2))
three_layer = list(itertools.combinations(test_list, 3))

node_range = test_list + two_layer + three_layer
"""
Change below to add a model
"""
models_to_run={'RF':0,'DN':0,'GBDT':1} # Define which models to run
classifiers={'DN':MLPClassifier(max_iter=200), 'RF':RandomForestClassifier(n_estimators=500), 'GBDT': GradientBoostingClassifier(n_estimators=500)}

varCV={'DN':{'n_jobs':-1,'verbose':1,'cv':None},
            'RF':{'n_jobs':-1,'verbose':1,'cv':None},
            'GBDT':{'n_jobs':None,'verbose':1,'cv':None}}

varargin = {'node_range':node_range, 'alpha_range_nn':alpha_range_nn,'subsample':subsample}



#%% function so save cc18_all_parameters file

# Empty dict to record optimal parameters
all_parameters = {model_name:{} for model_name,val in models_to_run.items() if val==1}
best_parameters = {model_name:{} for model_name,val in models_to_run.items() if val==1}
all_params = {model_name:{} for model_name,val in models_to_run.items() if val==1}

"""
Organize the data
"""
# Load data from CC18 data set suite
if (reload_data or 'dataset_name' not in locals()): # Load the data only if required (by reload_data or if it is not defined)
    X_data_list, y_data_list, dataset_name = load_cc18()

# Choose dataset indices
dataset_indices = [i for i in range(dataset_indices_max)]

# For each dataset, use randomized hyperparameter search to optimize parameters
for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)
        
    # Standart Scaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    p = X.shape[1]
    
    for model_name,val_run in models_to_run.items():
        if val_run==1:
            if model_name not in classifiers:
                raise ValueError('Model name is not defined in the classifiers dictionary')
            else:
                all_parameters, best_parameters,all_params=do_calcs_per_model(all_parameters,
                                                                              best_parameters, 
                                                                              all_params, model_name,
                                                                              varargin,varCV,classifiers,X,y,dataset_index)


save_best_parameters(save_methods,save_methods_rewrite,path_save,best_parameters)
