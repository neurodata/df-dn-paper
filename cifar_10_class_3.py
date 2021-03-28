from toolbox import *

import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
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


# prepare CIFAR data
def main():
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

    svm_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (svm)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for samples in samples_space:
            SVM = SVC()
            mean_accuracy = np.mean(
                [
                    run_rf_image_set(
                        SVM,
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
            svm_acc_vs_n.append(mean_accuracy)

    print("svm finished")
    write_result("3_class/svm.txt", svm_acc_vs_n)

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


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
