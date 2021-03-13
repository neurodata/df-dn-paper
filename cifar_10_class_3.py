from toolbox import *
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def write_result(filename, acc_ls):
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def load_result(filename):
    input = open(filename, "r")
    lines = input.readlines()
    ls = []
    for line in lines:
        ls.append(float(line.strip()))
    return ls


def produce_mean(acc_ls):
    ls_space = []
    for i in range(int(len(acc_ls) / 8)):
        ls = acc_ls[i * 8 : (i + 1) * 8]
        ls_space.append(ls)

    return np.mean(ls_space, axis=0)


# prepare CIFAR data
def main():
    names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    classes_space = list(combinations(nums, 3))[:45]

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

    naive_rf_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (naive_rf)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for samples in samples_space:
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            mean_accuracy = np.mean(
                [
                    run_rf_image_set(
                        RF,
                        cifar_train_images,
                        cifar_train_labels,
                        cifar_test_images,
                        cifar_test_labels,
                        samples,
                        classes,
                    )
                    for _ in range(1)
                ]
            )
            naive_rf_acc_vs_n.append(mean_accuracy)

    print("naive_rf finished")
    write_result("3_class/naive_rf.txt", naive_rf_acc_vs_n)

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cnn32_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
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
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image(
                        cnn32,
                        train_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_acc_vs_n.append(mean_accuracy)

    print("cnn32 finished")
    write_result("3_class/cnn32.txt", cnn32_acc_vs_n)

    cnn32_2l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
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
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image(
                        cnn32_2l,
                        train_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_2l_acc_vs_n.append(mean_accuracy)

    print("cnn32_2l finished")
    write_result("3_class/cnn32_2l.txt", cnn32_2l_acc_vs_n)

    cnn32_5l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
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
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image(
                        cnn32_5l,
                        train_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_5l_acc_vs_n.append(mean_accuracy)

    print("cnn32_5l finished")
    write_result("3_class/cnn32_5l.txt", cnn32_5l_acc_vs_n)

    # prepare CIFAR data
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    resnet18_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (resnet18)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
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
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image(
                        res,
                        train_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            resnet18_acc_vs_n.append(mean_accuracy)

    print("resnet18 finished")
    write_result("3_class/resnet18.txt", resnet18_acc_vs_n)

    # naive_rf_acc_vs_n = load_result("3_class/naive_rf.txt")
    # cnn32_acc_vs_n = load_result("3_class/cnn32.txt")
    # cnn32_2l_acc_vs_n = load_result("3_class/cnn32_2l.txt")
    # cnn32_5l_acc_vs_n = load_result("3_class/cnn32_5l.txt")
    # resnet18_acc_vs_n = load_result("3_class/resnet18.txt")

    samples_space = np.geomspace(10, 10000, num=8, dtype=int)
    fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
    plt.rcParams["figure.figsize"] = 13, 10
    plt.rcParams["font.size"] = 25
    plt.rcParams["legend.fontsize"] = 16.5
    plt.rcParams["legend.handlelength"] = 2.5
    plt.rcParams["figure.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.ylim([0, 1])
    ax.set_xlabel("Number of Train Samples", fontsize=18)
    ax.set_xscale("log")
    ax.set_xticks([i for i in list(samples_space)])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_ylabel("Accuracy", fontsize=18)

    ax.set_title(
        "3 classes classification",
        fontsize=18,
    )
    for i in range(45):
        ax.plot(
            samples_space,
            naive_rf_acc_vs_n[i * 8 : (i + 1) * 8],
            color="#e41a1c",
            alpha=0.1,
        )
        ax.plot(
            samples_space,
            cnn32_acc_vs_n[i * 8 : (i + 1) * 8],
            color="#377eb8",
            alpha=0.1,
        )
        ax.plot(
            samples_space,
            cnn32_2l_acc_vs_n[i * 8 : (i + 1) * 8],
            color="#4daf4a",
            alpha=0.1,
        )
        ax.plot(
            samples_space,
            cnn32_5l_acc_vs_n[i * 8 : (i + 1) * 8],
            color="#984ea3",
            alpha=0.1,
        )
        ax.plot(
            samples_space,
            resnet18_acc_vs_n[i * 8 : (i + 1) * 8],
            color="#ff7f00",
            alpha=0.1,
        )

    naive_rf_mean = produce_mean(naive_rf_acc_vs_n)
    cnn32_mean = produce_mean(cnn32_acc_vs_n)
    cnn32_2l_mean = produce_mean(cnn32_2l_acc_vs_n)
    cnn32_5l_mean = produce_mean(cnn32_5l_acc_vs_n)
    resnet18_mean = produce_mean(resnet18_acc_vs_n)

    ax.plot(
        samples_space,
        naive_rf_mean,
        color="#e41a1c",
        label="RF",
    )
    ax.plot(
        samples_space,
        cnn32_mean,
        color="#377eb8",
        label="CNN32",
    )
    ax.plot(
        samples_space,
        cnn32_2l_mean,
        color="#4daf4a",
        label="CNN32_2l",
    )
    ax.plot(
        samples_space,
        cnn32_5l_mean,
        color="#984ea3",
        label="CNN32_5l",
    )
    ax.plot(
        samples_space,
        resnet18_mean,
        color="#ff7f00",
        label="Resnet18",
    )

    plt.legend()
    plt.savefig("3_class/3 classes classification.png")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
