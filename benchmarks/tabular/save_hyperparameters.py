# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:19:19 2021

@author: noga mudrik
"""
import os
from os.path import exists
import ast
import pandas as pd
import json


# Save optimal parameters to txt file
def save_best_parameters(
    save_methods, save_methods_rewrite, path_save, best_parameters
):
    if save_methods["text_dict"]:
        if (
            os.path.exists(path_save + ".txt")
            and save_methods_rewrite["text_dict"] == 0
        ):
            file = open(path_save + ".txt", "r")
            contents = file.read()
            # display(contents)
            dictionary = ast.literal_eval(contents)
            best_parameters_to_save = {
                **dictionary,
                **best_parameters,
            }  # This will overwrite existing models but not models that were removed
        else:
            best_parameters_to_save = best_parameters
        with open(path_save + ".txt", "w") as f:
            f.write("%s\n" % best_parameters_to_save)

    if save_methods["json"]:
        if os.path.exists(path_save + ".json") and save_methods_rewrite["json"] == 0:
            with open(path_save + ".json", "r") as json_file:
                dictionary = json.load(json_file)
            best_parameters_to_save = {**dictionary, **best_parameters}
        else:
            best_parameters_to_save = best_parameters
        with open(path_save + ".json", "w") as fp:
            json.dump(best_parameters_to_save, fp)
    if save_methods["csv"]:
        df_new_data = pd.DataFrame(best_parameters_to_save)
        if os.path.exists(path_save + ".csv") and save_methods_rewrite["csv"] == 0:
            df_old = pd.read_csv(path_save + ".csv", index=False)
            df_to_save = pd.concat([df_new_data, df_old], 1, ignore_index=True)
        else:
            df_to_save = df_new_data
        df_to_save.to_csv(path_save + ".csv", index=False)


def open_data(path, format_file):
    if format_file == "text_dict":
        file = open(path + ".txt", "r")
        contents = file.read()
        # display(contents)
        dictionary = ast.literal_eval(contents)
        return dictionary
