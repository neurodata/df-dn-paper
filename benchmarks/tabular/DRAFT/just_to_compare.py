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


def open_data(path, format_file):
    if format_file == "text_dict":
        file = open(path + ".txt", "r")
        contents = file.read()
        # display(contents)
        dictionary = ast.literal_eval(contents)
        return dictionary


def mod_dict(res_dict, type_to_compare):
    tot_dict = {}
    for model_name in res_dict.keys():
        modified_subdict = {}
        for dataset in res_dict[model_name].keys():
            data_set_dict = res_dict[model_name][dataset]
            data_set_dict = {
                key: val
                for key, val in data_set_dict.items()
                if isinstance(val, type_to_compare)
            }
            modified_subdict[dataset] = data_set_dict
        tot_dict[model_name] = modified_subdict
    return tot_dict


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


def addi(
    total_len_sizes_and_data=total_len_sizes_and_data,
    path1="results/cc18_rf_kappa",
    path2="metrics/cc18_sample_sizes_new",
    name_model="RF",
    name_metric="cohen_kappa",
):
    file_sizes = open(path1 + ".txt", "r")
    cont_sizes = file_sizes.read()
    d = np.array(cont_sizes.split("\n"))
    d_2d = [[d_el.split(" ")] for d_el in d]
    dict_to_store = {}
    len_sample_sizes = 8
    for iterat_num, iterat in enumerate(
        range(0, total_len_sizes_and_data, len_sample_sizes)
    ):
        if iterat_num < 20:
            dict_to_store[iterat_num] = {}
            curr_d = d_2d[iterat : iterat + len_sample_sizes]
            key_iterat = list(full_dict[iterat_num].keys())
            for row_for_sample_size in range(len_sample_sizes):
                if len(key_iterat) > row_for_sample_size:
                    curr_key = key_iterat[row_for_sample_size]
                    dict_to_store[iterat_num][curr_key] = tuple(
                        curr_d[row_for_sample_size][0]
                    )

    return dict_to_store


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
