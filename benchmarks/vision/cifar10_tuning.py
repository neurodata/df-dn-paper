"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
"""
from toolbox import *

import argparse
import random
from sklearn.ensemble import RandomForestClassifier

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

def init_net(parameterization, model, classes):
  net = model
  return net # return untrained model


# Add parameters
def training_net(model, parameters, train_loader, device):
  # Training loop copied over from run_dn_image_es()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    epochs = parameters.get("epoch", 30)
    batch = 5
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if parameters.get("optimizer", "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=parameters.get("lr", 0.001), momentum=parameters.get("momentum", 0.9))
    if parameters.get("optimizer", "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=parameters.get("lr", 0.001))

    while True:  # loop over the dataset multiple times
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

    return (
        cohen_kappa_score(test_preds, test_labels)
    )

def run_dn_image_es(
    model,
    train_loader,
    valid_loader,
    test_loader,
    classes
):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """
    
    #init_net(parameters, classes)

    #training_net(model, parameters, train_data, train_labels, valid_data, valid_labels, parameters, device)

    # train_evaluate()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ax = AxClient(enforce_sequential_optimization=False)

    def train_evaluate(parameterization):
    # Ax primitive that initializes the training sequence --> Trains the model --> Calculates the evaluation metric
      untrained_net = init_net(parameterization, model, classes)
      trained_net = training_net(untrained_net, parameterization, train_loader, device)
      report(
        cohen_kappa=evaluate_net(trained_net, valid_loader, device)
    )
    
    ax.create_experiment(
    name="cifar10_experiment",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4],"log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "epoch", "type": "range", "bounds": [15, 40]},
        {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam"]}],
    objective_name="cohen_kappa",
    minimize=False)
    
    asha_scheduler = ASHAScheduler(
    max_t=30,
    grace_period=5,
    reduction_factor=2)

    algo = AxSearch(ax_client=ax)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization receives the
    # data for completed trials before creating more trials
    algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=1)
    tune.run(
        tune.with_parameters(train_evaluate),
        num_samples=5,
        metric="cohen_kappa",
        mode="max",
        search_alg=algo,
        verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        scheduler=asha_scheduler # To use GPU, specify: 
        #resources_per_trial={"gpu": 1, "cpu": 4}
    )

    best_parameters, values = ax.get_best_parameters()

    data = ax.experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = ax.experiment.arms_by_name[best_arm_name]

    best_objectives = np.array([[trial.objective_mean*100 for trial in ax.experiment.trials.values()]])
    
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
                batch = 5
            )
            arm, best_obj = run_dn_image_es(
                                            cnn32,
                                            train_loader,
                                            valid_loader,
                                            test_loader,
                                            classes
                                        )
            
            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset([
                                        train_loader.dataset.dataset, 
                                        valid_loader.dataset.dataset,
                                    ])
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set, 
                batch_size=60, 
                shuffle=True,
)
            model_retrain_aftertune = training_net(cnn32, arm.parameters, combined_train_valid_loader, device=device)

            test_accuracy = evaluate_net(model_retrain_aftertune, test_loader, 60, dev=device)

    print("cnn32 finished")
    print(test_accuracy)
    print(best_params)
    return best_params, best_objs, test_accuracy


def run_cnn32_2l():
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
            arm, best_obj = run_dn_image_es(
                                            cnn32_2l,
                                            train_loader,
                                            valid_loader,
                                            test_loader,
                                            classes
                                        )
            
            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset([
                                        train_loader.dataset.dataset, 
                                        valid_loader.dataset.dataset,
                                    ])
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set, 
                batch_size=60, 
                shuffle=True,
)
            model_retrain_aftertune = training_net(cnn32_2l, arm.parameters, combined_train_valid_loader, device=device)

            test_accuracy = evaluate_net(model_retrain_aftertune, test_loader, 60, dev=device)

    print("cnn32_2l finished")
    print(test_accuracy)
    print(best_params)
    return best_params, best_objs, test_accuracy


def run_cnn32_5l():
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
            arm, best_obj = run_dn_image_es(
                                            cnn32_5l,
                                            train_loader,
                                            valid_loader,
                                            test_loader,
                                            classes
                                        )
            
            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset([
                                        train_loader.dataset.dataset, 
                                        valid_loader.dataset.dataset,
                                    ])
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set, 
                batch_size=60, 
                shuffle=True,
)
            model_retrain_aftertune = training_net(cnn32_5l, arm.parameters, combined_train_valid_loader, device=device)

            test_accuracy = evaluate_net(model_retrain_aftertune, test_loader, 60, dev=device)

    print("cnn32_5l finished")
    print(test_accuracy)
    print(best_params)
    return best_params, best_objs, test_accuracy


def run_resnet18():
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
            arm, best_obj = run_dn_image_es(
                                            res,
                                            train_loader,
                                            valid_loader,
                                            test_loader,
                                            classes
                                        )
            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_set = torch.utils.data.ConcatDataset([
                                        train_loader.dataset.dataset, 
                                        valid_loader.dataset.dataset,
                                    ])
            combined_train_valid_loader = torch.utils.data.DataLoader(
                combined_train_valid_set, 
                batch_size=60, 
                shuffle=True,
)
            model_retrain_aftertune = training_net(res, arm.parameters, combined_train_valid_loader, device=device)

            resnet.eval()
            with torch.no_grad():
              test_accuracy = evaluate_net(model_retrain_aftertune, test_loader, 60, dev=device)


    print("resnet18 finished")
    print(test_accuracy)
    return best_params, best_objs, test_accuracy


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

    #run_naive_rf()

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    run_cnn32()
    # run_cnn32_2l()
    # run_cnn32_5l()

    # data_transforms = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )

    # run_resnet18()
