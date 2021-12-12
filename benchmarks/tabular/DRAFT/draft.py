# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 23:14:22 2021

@author: noga mudrik
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 05:08:11 2021

@author: noga mudrik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

#%% Read previous format to compare between dict form and array form


dict_ece_cohen = open_data("metrics/kappa_and_ece", "text_dict")
new_dict = {}
for key_met in dict_ece_cohen.keys():
    new_dict[key_met] = mod_dict(dict_ece_cohen[key_met], tuple)
cohens_results = new_dict["cohen_kappa"]
ece_results = new_dict["ece"]

full_dict = cohens_results["GBDT"]
#%%


path1 = "results/cc18_rf_kappa"

file = open(path1 + ".txt", "r")
# d = np.array(cont.split('\n'))
cont = file.read()
d = np.array(cont.split("\n"))
d_2d = [[d_el.split(" ")] for d_el in d]

data_rf_kappa = d_2d
total_len_sizes_and_data = len(d_2d) - 1


dict_to_store = addi(
    total_len_sizes_and_data=total_len_sizes_and_data,
    path1="results/cc18_rf_kappa",
    path2="metrics/cc18_sample_sizes_new",
    name_model="RF",
    name_metric="cohen_kappa",
)
new_dict["cohen_kappa"]["RF"] = dict_to_store

dict_to_store = addi(
    total_len_sizes_and_data=total_len_sizes_and_data,
    path1="results/cc18_dn_kappa",
    path2="metrics/cc18_sample_sizes_new",
    name_model="RF",
    name_metric="cohen_kappa",
)
new_dict["cohen_kappa"]["DN"] = dict_to_store

dict_to_store = addi(
    total_len_sizes_and_data=total_len_sizes_and_data,
    path1="results/cc18_rf_ece",
    path2="metrics/cc18_sample_sizes_new",
    name_model="RF",
    name_metric="cohen_kappa",
)
new_dict["ece"]["RF"] = dict_to_store

dict_to_store = addi(
    total_len_sizes_and_data=total_len_sizes_and_data,
    path1="results/cc18_dn_ece",
    path2="metrics/cc18_sample_sizes_new",
    name_model="RF",
    name_metric="cohen_kappa",
)
new_dict["ece"]["DN"] = dict_to_store
