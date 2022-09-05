"""
Coauthors: Michael Ainsworth
           Haoyin Xu
           Noga Mudrik
"""
# Imports

from sklearn.preprocessing import StandardScaler

from toolbox import *


# Define executrion variables (to save memory & execution time)
reload_data = False             # indicator of whether to upload the data again
return_default = False

"""
Hyperparameters options
"""

# General Parameters
dataset_indices_max = 72
max_shape_to_run = 1000 

# RF Hyper-Parameters
criterions         = ['gini', 'entropy']
max_features       = ['sqrt','log2',0.5,0.8,1]
max_depth          =  [1,3,7,8,9,10,15]
bootstrap          = [True,False]
n_estimators_range = [10,50,100,300]# [10,50,100,300]

# GBDT hyperparameters
eta                = [0.1,0.2,0.3,0.4,0.5,0.7,0.9]# default=0
gamma              = [0,0.3,0.2,0.1,0.2]          # default=0
subsample          = [0.5,0.6,0.7,1]              # default=1
sampling_method    = ['uniform','gradient_based'] # default= uniform
colsample_bynode   = [0.4,0.5,0.9,1]              # default = 1
lambda_vals        = [0.2,0.9,1,1.1]              # default = 1
alpha_vals         = [0.2,0.9,1,1.1]              # defaulte = 1
max_depth          = [2,3,4,6,10,12]


# TabNet
n_d                = [7,8,9,10,11]                # default=8
n_a                = [7,8,9,10,11]                # default=8
gamma_tabnet       = [1, 1.3, 1.5,1.8]            # default=1.3
n_shared           = [1,2,5]                      # default=2, Usual values range from 1 to 5
n_steps            = [2,3,5,7,8,10,15]            # default=3, usually between 3 and 10
lambda_sparse      = [1e-4, 1e-3, 1e-2, 0.1]      # default = 1e-3
momentum           = [0.01, 0.02,0.05,0.1,0.4]                 

# Saving parameters
path_save = "metrics/cc18_all_parameters"
path_save_dict_data_indices = "metrics/dict_data_indices"
save_methods = {"text_dict": 0, "csv": 0, "json": 1}
save_methods_rewrite = {"text_dict": 0, "csv": 0, "json": 1}

if return_default:
    (      
        dataset_indices_max,
        max_shape_to_run,
        models_to_run,
        subsample,
        ) = return_to_default()

"""
Models:
"""

"""
Change below to add a model
 
define which models to run
"""
models_to_run = {
    "RF": 1,
    "DN": 1,
    "GBDT": 1,
} 
models_to_scale = {
    "RF": 0,
    "DN": 1,
    "GBDT": 0,
} 

classifiers = {
    "DN": TabNetClassifier(),
    "RF": RandomForestClassifier(),
    "GBDT": xgb.XGBClassifier(booster="gbtree", base_score=0.5),
}


varargin = {
    "DN":
        {"n_d": n_d, "n_a": n_a, "gamma": gamma_tabnet,"n_shared": n_shared,'momentum': momentum,
         'lambda_sparse':lambda_sparse , 'n_steps':n_steps},
   "GBDT":
        { "subsample": subsample, 'alpha':alpha_vals,  'lambda': lambda_vals,
         'colsample_bynode':colsample_bynode, 'sampling_method':sampling_method,'gamma':gamma,'eta':eta,'max_depth':max_depth}, 
    "RF":
        {'n_estimators':n_estimators_range, 'criterion':criterions,'max_depth':max_depth,'bootstrap':bootstrap}
}

varCV = {
    "DN": {"n_jobs": -1, "verbose": 1},
    "RF": {"n_jobs": -1, "verbose": 1},
    "GBDT": {"n_jobs": -1, "verbose": 1},
}

save_vars_to_dict(
    classifiers,
    varargin,
    reload_data = reload_data,
    dataset_indices_max = dataset_indices_max,
    max_shape_to_run = max_shape_to_run,
    subsample = subsample,
    path_to_save = "metrics/dict_parameters.json",
)


"""
function to save cc18_all_parameters file
Empty dict to record optimal parameters
"""

all_parameters = {
    model_name: {} for model_name, val in models_to_run.items() if val == 1
}
best_parameters = {
    model_name: {} for model_name, val in models_to_run.items() if val == 1
}
all_params = {model_name: {} for model_name, val in models_to_run.items() if val == 1}


calc_times = {model_name: {int(dataset_index): {} for dataset_index in np.arange(dataset_indices_max)}
    for model_name in models_to_run.keys()
}

"""
Organize the data
"""
# Load data from CC18 data set suite
if (
    reload_data or "dataset_name" not in locals()
):  # Load the data only if required (by reload_data or if it is not defined)
    X_data_list, y_data_list, dataset_name = load_cc18()

"""
Choose dataset indices
"""

dataset_indices = list(range(dataset_indices_max))
dict_data_indices = {dataset_ind: {} for dataset_ind in dataset_indices}
# ss_inds_full is a dictionary of the indices taken for different datasets
ss_inds_full = {} 


"""
For each dataset, use randomized hyperparameter search to optimize parameters
"""

for dataset_index, dataset in enumerate(dataset_indices):

    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    """
    If data set has over 10000 samples, resample to contain 10000
    """

    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y, max_shape_to_run)

    """
    Split to train, val and test by ratio of 2:1:1
    """
    np.random.seed(dataset_index)
    dict_data_indices = find_indices_train_val_test(
        y, dict_data_indices=dict_data_indices, dataset_ind=dataset_index
    )
    train_indices = dict_data_indices[dataset_index]["train"]
    val_indices = dict_data_indices[dataset_index]["val"]


    p = X.shape[1]

    X_train = X[train_indices]
    y_train = y[train_indices]
    training_sample_sizes = np.geomspace(
        len(np.unique(y_train)) * 5, X_train.shape[0], num=8, dtype=int
    )
    ss_inds = random_sample_new(
        X_train, y_train, training_sample_sizes
    )
    ss_inds_full[dataset_index] = ss_inds

    for sample_size_index, sample_size in enumerate(training_sample_sizes):
        print(np.array(ss_inds[sample_size_index], dtype = int))
        #print(train_indices)
        train_indices = np.array(train_indices)
        train_indices_pers = train_indices[np.array(ss_inds[sample_size_index] , dtype = int).astype(int)]
        
        for model_name, val_run in models_to_run.items():
            if val_run == 1:
                if model_name not in classifiers:
                    raise ValueError(
                        "Model name is not defined in the classifiers dictionary"
                    )
                else:
                    if models_to_scale[model_name ]:
                        """
                        Standart Scaling
                        """
                        scaler = StandardScaler()
                        scaler.fit(X)
                        X = scaler.transform(X)
                        
                    all_parameters, best_parameters, all_params,calc_time = do_calcs_per_model(
                        all_parameters,
                        best_parameters,
                        all_params,
                        model_name,
                        varargin,
                        classifiers,
                        X,
                        y,
                        dataset_index,
                        train_indices_pers,
                        val_indices,
                        p,
                        varCV,
                        sample_size_index,
                                              
                    )

                    calc_times[model_name][int(dataset_index)][int(sample_size_index)] = calc_time

    save_best_parameters(save_methods, save_methods_rewrite, path_save, best_parameters)

    save_best_parameters(
        save_methods, save_methods_rewrite, path_save_dict_data_indices, dict_data_indices
    )
    save_best_parameters(
        save_methods, save_methods_rewrite, "metrics/varied_size_dict.json", ss_inds_full
    )
    save_best_parameters(
        save_methods, save_methods_rewrite, "metrics/tuning_times.json", calc_times
    )
    save_best_parameters(
        save_methods, save_methods_rewrite, "metrics/tuning_times.json", calc_times
    )
