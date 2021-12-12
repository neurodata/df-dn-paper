# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 01:28:23 2021

@author: noga mudrik
"""
# Analyze results
from save_hyperparameters import *
import ast
import matplotlib.pyplot as plt

if not os.path.exists(figspath):
    os.makedirs(figspath)
import matplotlib.patches as mpatches

path_save_train = "results/times_train"
path_save_test = "results/times_test"
figspath = "results/figs_results"


def plot_box(
    df_train, df_test, colors=["c", "m"], ax=None, model_name=None, xlabel="Data Size"
):
    df_train.boxplot(ax=ax, color=colors[0])
    df_test.boxplot(ax=ax, color=colors[1])
    red_patch = mpatches.Patch(color=colors[0], label="Train")
    blue_patch = mpatches.Patch(color=colors[1], label="Test")
    plt.legend(handles=[red_patch, blue_patch])
    model_name = model_name
    ax.set_title("Train vs Test times %s" % model_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Running time [s]")


def plot_times_fill(df, color, ax=None, label=None, title=None, xlabel="Data Size"):
    list_vals = []
    data_size_list = []
    for col in df.columns:
        list_vals.extend(df[col].to_list())
        data_size_list.extend([col] * len(df[col]))
    ax.scatter(data_size_list, list_vals, marker=".", color=color)
    ax.fill_between(df.columns, df.min(), df.max(), alpha=0.2, color=color)
    df.mean().plot(color=color, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Running time [s]")
    ax.set_title(title)


dictionary_train = open_data(path_save_train, "text_dict")
dictionary_train = {
    key: {key2: np.array(value) for key2, value in inner_res.items()}
    for key, inner_res in dictionary_train.items()
}

dictionary_test = open_data(path_save_test, "text_dict")
dictionary_test = {
    key: {key2: np.array(value) for key2, value in inner_res.items()}
    for key, inner_res in dictionary_test.items()
}
available_colors = np.array(
    ["c", "blue", "m", "purple", "green", "lightgreen", "crimson", "red"]
).reshape(-1, 2)

fig1, axs1 = plt.subplots(1, len(dictionary_train.keys()))
if not isinstance(axs1, np.ndarray):
    axs1 = [axs1]
fig2, axs2 = plt.subplots(1, len(dictionary_train.keys()))
if not isinstance(axs2, np.ndarray):
    axs2 = [axs2]
# Each sub-dict is the 5-fold cv measurments for a different size of data.
for model_num, model_name in enumerate(dictionary_train.keys()):
    # Boxplot
    df_train = pd.DataFrame(dictionary_train[model_name])
    df_test = pd.DataFrame(dictionary_test[model_name])
    plot_box(
        df_train,
        df_test,
        colors=available_colors[model_num],
        ax=axs1[model_num],
        model_name=model_name,
        xlabel="Data Size",
    )
    plot_times_fill(
        df_train, available_colors[model_num][0], axs2[model_num], "Train", model_name
    )
    plot_times_fill(
        df_test, available_colors[model_num][1], axs2[model_num], "Test", model_name
    )
    axs2[model_num].legend()
plt.figure(fig1)
plt.savefig(figspath + "/times_box_plot.png")
plt.close(fig1)
fig2
plt.savefig(figspath + "/plot_times.png")
plt.close(fig2)

# Plot
