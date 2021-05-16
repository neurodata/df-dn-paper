"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
"""
from toolbox import *

import argparse
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# prepare CIFAR data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-s", help="computation speed")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"

    if args.s == "h":
        # High speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_time.txt"))
        ratio = 1.0
    elif args.s == "l":
        # Low speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_time_lc.txt"))
        ratio = 0.11 / 0.9
    else:
        raise Exception("Wrong configurations for time calibration.")

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cnn32_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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

            start_time = time.perf_counter()
            time_limit = rf_times[i]
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
                    run_dn_image_set(
                        cnn32,
                        train_loader,
                        test_loader,
                        start_time=start_time,
                        time_limit=time_limit,
                        ratio=ratio,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_acc_vs_n.append(mean_accuracy)

    print("cnn32 finished")
    if args.s == "h":
        write_result(prefix + "cnn32_st.txt", cnn32_acc_vs_n)
    elif args.s == "l":
        write_result(prefix + "cnn32_sc.txt", cnn32_acc_vs_n)

    cnn32_2l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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

            start_time = time.perf_counter()
            time_limit = rf_times[i]
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
                    run_dn_image_set(
                        cnn32_2l,
                        train_loader,
                        test_loader,
                        start_time=start_time,
                        time_limit=time_limit,
                        ratio=ratio,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_2l_acc_vs_n.append(mean_accuracy)

    print("cnn32_2l finished")
    if args.s == "h":
        write_result(prefix + "cnn32_2l_st.txt", cnn32_2l_acc_vs_n)
    elif args.s == "l":
        write_result(prefix + "cnn32_2l_sc.txt", cnn32_2l_acc_vs_n)

    cnn32_5l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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

            start_time = time.perf_counter()
            time_limit = rf_times[i]
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
                    run_dn_image_set(
                        cnn32_5l,
                        train_loader,
                        test_loader,
                        start_time=start_time,
                        time_limit=time_limit,
                        ratio=ratio,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_5l_acc_vs_n.append(mean_accuracy)

    print("cnn32_5l finished")
    if args.s == "h":
        write_result(prefix + "cnn32_5l_st.txt", cnn32_5l_acc_vs_n)
    elif args.s == "l":
        write_result(prefix + "cnn32_5l_sc.txt", cnn32_5l_acc_vs_n)

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
        for i, samples in enumerate(samples_space):
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

            start_time = time.perf_counter()
            time_limit = rf_times[i]
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
                    run_dn_image_set(
                        res,
                        train_loader,
                        test_loader,
                        start_time=start_time,
                        time_limit=time_limit,
                        ratio=ratio,
                    )
                    for _ in range(1)
                ]
            )
            resnet18_acc_vs_n.append(mean_accuracy)

    print("resnet18 finished")
    if args.s == "h":
        write_result(prefix + "resnet18_st.txt", resnet18_acc_vs_n)
    elif args.s == "l":
        write_result(prefix + "resnet18_sc.txt", resnet18_acc_vs_n)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
