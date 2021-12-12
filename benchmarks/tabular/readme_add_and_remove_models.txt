
*** Adding a model ***

In cc18_hyperparameter.py, under the #%% Models section:

a) In the dictionary 'models_to_run' - add the name (str) of the model as a key and a binary value (1/0) as a value. this binary value indicates whether to run the model in the current execution.
b) In the dictionary 'classifiers' - Add the name (str) of the model (the same name used before) as a key and the relevant classifier (e.g., MLPClassifier(max_iter=200) for DN).
c) In the dictionary 'varCV' -  Add the name (str) of the model (the same name used before) as a key and a dict as a value. The dict should include the relevant cv parameters (n_jobs, verbose, cv).


*** Removing a model from current run ***
In order to disable a specific model in a specific execution, just change its value in 'models_to_run' to 0.


*** Comparing models ***
If you want to compare several models, set their value to 1 in 'model_to_run' dictionary


*** Go to default values***
Just run the following function in the command window:
nodes_combination,dataset_indices_max,max_shape_to_run,models_to_run,subsample,alpha_range_nn = return_to_default()


*** Save methods ***
3 options are available - you can choose which one (or more) you want to use by setting their value to 1 in the 'save_methods' dictionary.
The options are:
1) .txt file containing a dict of the best parameters.
2) .csv file
3) Json file containing the dictionary


