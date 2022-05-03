"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Adway Kanhere
"""
from toolbox import *
from fsdk18preprocess import *
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.models as models
import warnings
import random
import ax
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.service.ax_client import AxClient
import logging
from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.

warnings.filterwarnings("ignore")

# Ax function to initialize the model
def init_net(model, classes, parameters):
    if model == "cnn32":
        net = SimpleCNN32Filter(len(classes))
    elif model == "cnn32_2l":
        net = SimpleCNN32Filter2Layers(len(classes))
    elif model == "cnn32_5l":
        net = SimpleCNN32Filter5Layers(len(classes))
    elif model == "resnet":
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(classes))
    return net  # return untrained model


# Add parameters
def training_net(model, parameters, train_data, train_labels):
    # Training loop copied over from run_dn_image_es()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    epochs = parameters.get("epoch", 30)
    batch = 60
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if parameters.get("optimizer", "SGD"):
        optimizer = optim.SGD(
            model.parameters(),
            lr=parameters.get("lr", 0.001),
            momentum=parameters.get("momentum", 0.9),
        )
    if parameters.get("optimizer", "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=parameters.get("lr", 0.001))

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        for i in range(0, len(train_data), batch):
            # get the inputs
            inputs = train_data[i : i + batch].to(dev)
            labels = train_labels[i : i + batch].to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# Add parameters
def evaluate_net(model, given_data, given_labels, batch, dev):

    # Function that evaluates the model and return the desired metric to optimize. Copied over from run_dn_image_es()

    model.eval()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(given_data), batch):
            inputs = given_data[i : i + batch].to(dev)
            labels = given_labels[i : i + batch].to(dev)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if i == 0:
                test_probs = test_prob.tolist()
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    test_labels = np.array(given_labels.tolist())
    return accuracy_score(test_preds, test_labels)


def evaluate_net_final(model, given_data, given_labels, batch, dev):

    # Function that evaluates the cohen kappa.

    model.eval()
    start_time = time.perf_counter()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(given_data), batch):
            inputs = given_data[i : i + batch].to(dev)
            labels = given_labels[i : i + batch].to(dev)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if i == 0:
                test_probs = test_prob.tolist()
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    test_labels = np.array(given_labels.tolist())
    end_time = time.perf_counter()
    test_time = end_time - start_time
    return (
        cohen_kappa_score(test_preds, test_labels),
        get_ece(test_probs, test_preds, test_labels),
        test_time,
    )


def run_dn_image_es(
    model,
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    test_data,
    test_labels,
    classes,
):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """

    # init_net(parameters, classes)
    # train_evaluate()
    batch = 60

    ax = AxClient(enforce_sequential_optimization=False)

    def train_evaluate(parameterization):
        # Ax primitive that initializes the training sequence --> Trains the model --> Calculates the evaluation metric
        untrained_net = init_net(model, classes, parameterization)
        trained_net = training_net(
            untrained_net, parameterization, train_data, train_labels
        )
        report(
            accuracy=evaluate_net(trained_net, valid_data, valid_labels, batch, device)
        )

    ax.create_experiment(
        name="fsdk18_experiment",
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "epoch", "type": "range", "bounds": [15, 40]},
            {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam"]},
        ],
        objective_name="accuracy",
        minimize=False,
    )

    asha_scheduler = ASHAScheduler(max_t=30, grace_period=5, reduction_factor=2)

    algo = AxSearch(ax_client=ax)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization receives the
    # data for completed trials before creating more trials
    algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=3)
    tune.run(
        train_evaluate,
        num_samples=20,
        metric="accuracy",
        mode="max",
        search_alg=algo,
        verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        scheduler=asha_scheduler,  # To use GPU, specify: resources_per_trial={"gpu": 1}.
    )

    data = ax.experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df["mean"] == df["mean"].max()].values[0]
    best_arm = ax.experiment.arms_by_name[best_arm_name]

    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in ax.experiment.trials.values()]]
    )

    return best_arm, best_objectives


def run_naive_rf():
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            # train data
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            cohen_kappa, ece, train_time, test_time = run_rf_image_set(
                RF,
                fsdk18_train_images,
                fsdk18_train_labels,
                fsdk18_test_images,
                fsdk18_test_labels,
                samples,
                classes,
            )
            naive_rf_kappa.append(cohen_kappa)
            naive_rf_ece.append(ece)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

    print("naive_rf finished")
    write_result(prefix + "naive_rf_kappa.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    cnn32_tune_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32)
        for classes in classes_space:
            # train data
            cnn32 = SimpleCNN32Filter(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_final = SimpleCNN32Filter(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
                device=device,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_kappa.append(final_ck)
            cnn32_ece.append(final_ece)
            cnn32_tune_time.append(tune_time)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result("cnn32_bestparams.txt", best_params)
    write_result("cnn32_kappa.txt", cnn32_kappa)
    write_result("cnn32_ece.txt", cnn32_ece)
    write_result("cnn32_tune_time.txt", cnn32_tune_time)
    write_result("cnn32_train_time.txt", cnn32_train_time)
    write_result("cnn32_test_time.txt", cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_tune_time = []
    cnn32_2l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for classes in classes_space:
            # train data
            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_2l",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_2l_final = SimpleCNN32Filter2Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_2l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
                device=device,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_2l_kappa.append(final_ck)
            cnn32_2l_ece.append(final_ece)
            cnn32_2l_tune_time.append(tune_time)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result("cnn32_2l_bestparams.txt", best_params)
    write_result("cnn32_2l_kappa.txt", cnn32_2l_kappa)
    write_result("cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result("cnn32_2l_tune_time.txt", cnn32_2l_tune_time)
    write_result("cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result("cnn32_2l_test_time.txt", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_tune_time = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for classes in classes_space:
            # train data
            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_5l",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_5l_final = SimpleCNN32Filter5Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_5l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
                device=device,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_5l_kappa.append(final_ck)
            cnn32_5l_ece.append(final_ece)
            cnn32_5l_tune_time.append(tune_time)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result("cnn32_5l_bestparams.txt", best_params)
    write_result("cnn32_5l_kappa.txt", cnn32_5l_kappa)
    write_result("cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result("cnn32_5l_tune_time.txt", cnn32_5l_tune_time)
    write_result("cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result("cnn32_5l_test_time.txt", cnn32_5l_test_time)


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_tune_time = []
    resnet18_train_time = []
    resnet18_test_time = []
    best_objs = []
    best_params = []

    for samples in samples_space:
        # cohen_kappa vs num training samples (resnet18)
        for classes in classes_space:
            # Initialize model
            resnet18 = models.resnet18(pretrained=True)
            num_ftrs = resnet18.fc.in_features
            resnet18.fc = nn.Linear(num_ftrs, len(classes))
            # train data
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            # need to duplicate channel because batch norm cant have 1 channel images

            train_images = torch.cat((train_images, train_images, train_images), dim=1)
            valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)
            test_images = torch.cat((test_images, test_images, test_images), dim=1)

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "resnet",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            resnet_final = models.resnet18(pretrained=True)
            num_ftrs = resnet_final.fc.in_features
            resnet_final.fc = nn.Linear(num_ftrs, len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                resnet_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
                device=device,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            resnet18_kappa.append(final_ck)
            resnet18_ece.append(final_ece)
            resnet18_tune_time.append(tune_time)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result("resnet18_bestparams.txt", best_params)
    write_result("resnet18_kappa.txt", resnet18_kappa)
    write_result("resnet18_ece.txt", resnet18_ece)
    write_result("resnet18_tune_time.txt", resnet18_tune_time)
    write_result("resnet18_train_time.txt", resnet18_train_time)
    write_result("resnet18_test_time.txt", resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-f", help="feature type")
    parser.add_argument("-data", help="audio files location")
    parser.add_argument("-labels", help="labels file location")
    args = parser.parse_args()
    n_classes = int(args.m)
    feature_type = str(args.f)

    # Preprocess and subset the data
    path_recordings, labels_chosen, get_labels = preprocessdataset(
        str(args.data), str(args.labels)
    )

    # data is normalized upon loading
    # load dataset
    x_spec, y_number = load_fsdk18(
        path_recordings, labels_chosen, get_labels, feature_type
    )
    nums = list(range(18))
    samples_space = np.geomspace(10, 450, num=6, dtype=int)
    # define path, samples space and number of class combinations
    if feature_type == "melspectrogram":
        prefix = args.m + "_class_mel/"
    elif feature_type == "spectrogram":
        prefix = args.m + "_class/"
    elif feature_type == "mfcc":
        prefix = args.m + "_class_mfcc/"

    # create list of classes with const random seed
    random.Random(5).shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # scale the data
    x_spec = scale(x_spec.reshape(5400, -1), axis=1).reshape(5400, 32, 32)
    y_number = np.array(y_number)

    # need to take train/valid/test equally from each class
    trainx, testx, trainy, testy = train_test_split(
        x_spec,
        y_number,
        shuffle=True,
        test_size=0.50,
        train_size=0.50,
        stratify=y_number,
    )

    # 3000 samples, 80% train is 2400 samples, 20% test
    fsdk18_train_images = trainx.reshape(-1, 32 * 32)
    fsdk18_train_labels = trainy.copy()
    # reshape in 2d array
    fsdk18_test_images = testx.reshape(-1, 32 * 32)
    fsdk18_test_labels = testy.copy()

    run_naive_rf()
    run_cnn32()
    run_cnn32_2l()
    run_cnn32_5l()
    run_resnet18()
