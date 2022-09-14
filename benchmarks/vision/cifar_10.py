"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Adway Kanhere
"""
from toolbox import *

import argparse
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from copy import deepcopy
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import warnings
import ax
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ax function to initialize the model
def init_net(model, classes, parameters):
    if model == "cnn32":
        net = SimpleCNN32Filter(len(classes))
    elif model == "cnn32_2l":
        net = SimpleCNN32Filter2Layers(len(classes))
    elif model == "cnn32_5l":
        net = SimpleCNN32Filter5Layers(len(classes))
    elif model == "resnet18":
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(classes))
    return net  # return untrained model


# Add parameters
def train_net(model, parameters, train_loader):
    # Training loop copied over from run_dn_image_es()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    epochs = parameters.get("epoch", 30)

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
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# Add parameters
def evaluate_net(model, test_loader, dev):
    # Function that evaluates the model and return the desired metrics
    model.eval()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            test_labels = np.concatenate((test_labels, labels.tolist()))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if first:
                test_probs = test_prob.tolist()
                first = False
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    return accuracy_score(test_preds, test_labels)


def evaluate_net_final(model, test_loader, dev):
    # Function that evaluates the cohen kappa & ece.
    model.eval()
    start_time = time.perf_counter()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            test_labels = np.concatenate((test_labels, labels.tolist()))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if first:
                test_probs = test_prob.tolist()
                first = False
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
    train_loader,
    valid_loader,
    classes,
):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """
    ax = AxClient(enforce_sequential_optimization=False)

    def train_evaluate(parameterization):
        # Ax primitive that trains the model --> calculates the evaluation metric
        untrained_net = init_net(model, classes, parameterization)
        trained_net = train_net(untrained_net, parameterization, train_loader)
        report(accuracy=evaluate_net(trained_net, valid_loader, device))

    ax.create_experiment(
        name="cifar_experiment",
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "epoch", "type": "range", "bounds": [15, 40]},
            {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam"]},
        ],
        objective_name="accuracy",
        minimize=False,
    )

    asha_scheduler = ASHAScheduler(grace_period=5, reduction_factor=2)

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
        scheduler=asha_scheduler,  # To use GPU, specify:
        resources_per_trial={"gpu": 1},
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
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            cohen_kappa, ece, train_time, test_time = run_rf_image_set(
                RF,
                cifar_train_images,
                cifar_train_labels,
                cifar_test_images,
                cifar_test_labels,
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
    cnn32_tune_time = []
    cnn32_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32)
        for classes in classes_space:

            # Create loaders
            train_loader, valid_loader = create_loaders_ra(
                cifar_train_labels.copy(),
                cifar_valid_labels.copy(),
                classes,
                deepcopy(cifar_train_set),
                deepcopy(cifar_valid_set),
                samples,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32",
                train_loader,
                valid_loader,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            cnn32_final = SimpleCNN32Filter2Layers(len(classes))

            train_valid_loader, test_loader = create_loaders_ra(
                cifar_train_valid_labels.copy(),
                cifar_test_labels.copy(),
                classes,
                deepcopy(cifar_train_valid_set),
                deepcopy(cifar_test_set),
                samples,
            )

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_final,
                arm.parameters,
                train_valid_loader,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
            )

            cnn32_kappa.append(final_ck)
            cnn32_ece.append(final_ece)
            cnn32_tune_time.append(tune_time)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

            write_result(prefix + "cnn32_bestparams.txt", best_params)
            write_result(prefix + "cnn32_kappa.txt", cnn32_kappa)
            write_result(prefix + "cnn32_ece.txt", cnn32_ece)
            write_result(prefix + "cnn32_tune_time.txt", cnn32_tune_time)
            write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
            write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)
    print("cnn32 finished")


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

            cnn32_2l_final = SimpleCNN32Filter2Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_2l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
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
    write_result(prefix + "cnn32_2l_bestparams.txt", best_params)
    write_result(prefix + "cnn32_2l_kappa.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_tune_time.txt", cnn32_2l_tune_time)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_tune_time = []
    cnn32_5l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for classes in classes_space:

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

            cnn32_5l_final = SimpleCNN32Filter2Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_5l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
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
    write_result(prefix + "cnn32_5l_bestparams.txt", best_params)
    write_result(prefix + "cnn32_5l_kappa.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_tune_time.txt", cnn32_5l_tune_time)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_tune_time = []
    resnet18_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (resnet18)
        for classes in classes_space:
            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                cifar_train_set.copy(),
                cifar_train_labels.copy(),
                cifar_test_set.copy(),
                cifar_test_labels.copy(),
                samples,
                classes,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "resnet18",
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

            resnet_final = models.resnet18(pretrained=True)
            num_ftrs = resnet_final.fc.in_features
            resnet_final.fc = nn.Linear(num_ftrs, len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                resnet18_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
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
    write_result(prefix + "resnet18_bestparams.txt", best_params)
    write_result(prefix + "resnet18_kappa.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece.txt", resnet18_ece)
    write_result(prefix + "resnet18_tune_time.txt", resnet18_tune_time)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_10.py -m 3
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(10, 10000, num=8, dtype=int)

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    # scale = np.mean(np.arange(0, 256))
    # normalize = lambda x: (x - scale) / scale

    # # train data
    # cifar_train_set = datasets.CIFAR10(
    #     root="./", train=True, download=True, transform=None
    # )
    # cifar_train_images = normalize(cifar_train_set.data)
    # cifar_train_labels = np.array(cifar_train_set.targets)

    # # test data
    # cifar_test_set = datasets.CIFAR10(
    #     root="./", train=False, download=True, transform=None
    # )
    # cifar_test_images = normalize(cifar_test_set.data)
    # cifar_test_labels = np.array(cifar_test_set.targets)

    # cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
    # cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)

    # run_naive_rf()

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # train data
    cifar_train_set = datasets.CIFAR10(
        root="./", train=True, download=True, transform=data_transforms
    )

    # test data
    cifar_test_set = datasets.CIFAR10(
        root="./", train=False, download=True, transform=data_transforms
    )

    # Combine all data into whole set
    cifar_whole_images = np.concatenate((cifar_train_set.data, cifar_test_set.data))
    cifar_whole_labels = np.concatenate(
        (np.array(cifar_train_set.targets), np.array(cifar_test_set.targets))
    )

    # Separate whole set into training set & valid set & test set with 2:1:1 ratio
    (
        cifar_train_valid_images,
        cifar_test_images,
        cifar_train_valid_labels,
        cifar_test_labels,
    ) = train_test_split(
        cifar_whole_images,
        cifar_whole_labels,
        test_size=0.25,
        stratify=cifar_whole_labels,
    )
    (
        cifar_valid_images,
        cifar_train_images,
        cifar_valid_labels,
        cifar_train_labels,
    ) = train_test_split(
        cifar_train_valid_images,
        cifar_train_valid_labels,
        test_size=0.67,
        stratify=cifar_train_valid_labels,
    )

    # Create new datasets
    cifar_train_valid_set = deepcopy(cifar_train_set)
    cifar_train_valid_set.data = cifar_train_valid_images
    cifar_train_valid_set.targets = cifar_train_valid_labels

    cifar_valid_set = deepcopy(cifar_train_set)
    cifar_valid_set.data = cifar_valid_images
    cifar_valid_set.targets = cifar_valid_labels

    cifar_test_set = deepcopy(cifar_train_set)
    cifar_test_set.data = cifar_test_images
    cifar_test_set.targets = cifar_test_labels

    cifar_train_set.data = cifar_train_images
    cifar_train_set.targets = cifar_train_labels

    run_cnn32()
    # run_cnn32_2l()
    # run_cnn32_5l()

    # data_transforms = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )

    # init_dataset(data_transforms)
    # run_resnet18()
