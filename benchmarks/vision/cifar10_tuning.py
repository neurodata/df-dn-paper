from toolbox import *

import argparse
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import ax
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.service.ax_client import AxClient
import logging
import warnings

from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.

warnings.filterwarnings("ignore")


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
def training_net(model, parameters, train_loader):
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

    for _ in range(epochs):  # loop over the dataset multiple times
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
    # Function that evaluates the model and return the desired metric to optimize. Copied over from run_dn_image_es()
    model.eval()
    first = True
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
    # Function that evaluates the model and return the desired metric to optimize. Copied over from run_dn_image_es()
    model.eval()
    start_time = time.perf_counter()
    first = True
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
    end_time = time.perf_counter()
    test_time = end_time - start_time
    return (
        cohen_kappa_score(test_preds, test_labels),
        get_ece(test_probs, test_preds, test_labels),
        test_time,
    )


def run_dn_image_es(model, train_loader, valid_loader, classes):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ax = AxClient(enforce_sequential_optimization=False)

    def train_evaluate(parameterization):
        # Ax primitive that initializes the training sequence --> Trains the model --> Calculates the evaluation metric
        untrained_net = init_net(model, classes, parameterization)
        trained_net = training_net(untrained_net, parameterization, train_loader)
        return evaluate_net(trained_net, valid_loader, device)

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "epoch", "type": "range", "bounds": [15, 40]},
            {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam"]},
        ],
        evaluation_function=train_evaluate,
        objective_name="accuracy",
    )

    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df["mean"] == df["mean"].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]

    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in experiment.trials.values()]]
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
    cnn32_tune_time = []
    cnn32_train_time = []
    cnn32_test_time = []
    best_objs = []
    best_params = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32 = SimpleCNN32Filter(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32", train_loader, valid_loader, classes
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset(
                [
                    train_loader.dataset,
                    valid_loader.dataset,
                ]
            )
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set,
                batch_size=40,
                shuffle=True,
            )

            cnn32_final = SimpleCNN32Filter(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_final, arm.parameters, combined_train_valid_loader
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
    cnn32_2l_tune_time = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    best_objs = []
    best_params = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_2l", train_loader, valid_loader, classes
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset(
                [
                    train_loader.dataset,
                    valid_loader.dataset,
                ]
            )
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set,
                batch_size=40,
                shuffle=True,
            )
            cnn32_2l_final = SimpleCNN32Filter2Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_2l_final, arm.parameters, combined_train_valid_loader
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
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
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es("cnn32_5l", train_loader, valid_loader)
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset(
                [
                    train_loader.dataset,
                    valid_loader.dataset,
                ]
            )
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set,
                batch_size=40,
                shuffle=True,
            )

            cnn32_5l_final = SimpleCNN32Filter5Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_5l_final, arm.parameters, combined_train_valid_loader
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
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
    for classes in classes_space:

        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            res = models.resnet18(pretrained=True)
            num_ftrs = res.fc.in_features
            res.fc = nn.Linear(num_ftrs, len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "resnet", train_loader, valid_loader, classes
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset(
                [
                    train_loader.dataset,
                    valid_loader.dataset,
                ]
            )
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set,
                batch_size=40,
                shuffle=True,
            )
            res_final = models.resnet18(pretrained=True)
            num_ftrs = res_final.fc.in_features
            res_final.fc = nn.Linear(num_ftrs, len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                res_final, arm.parameters, combined_train_valid_loader
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
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
    scale = np.mean(np.arange(0, 256))
    normalize = lambda x: (x - scale) / scale

    # train data
    cifar_trainset = datasets.CIFAR10(
        root="./", train=True, download=True, transform=None
    )
    cifar_train_images = normalize(cifar_trainset.data)
    cifar_train_labels = np.array(cifar_trainset.targets)

    # test data
    cifar_testset = datasets.CIFAR10(
        root="./", train=False, download=True, transform=None
    )
    cifar_test_images = normalize(cifar_testset.data)
    cifar_test_labels = np.array(cifar_testset.targets)

    cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
    cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)

    run_naive_rf()

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    run_cnn32()
    run_cnn32_2l()
    run_cnn32_5l()

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    run_resnet18()
