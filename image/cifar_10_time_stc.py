"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
"""
import timeit
import torch
import argparse
import random
import numpy as np
from toolbox import write_result, combinations_45


def time_cnn32(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR10(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR10(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32 = SimpleCNN32Filter(len({}))
train_loader, test_loader = create_loaders_set(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image(
    cnn32,
    train_loader,
    test_loader,
)
""".format(
                classes, classes, samples
            )
            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            cnn32_time_vs_n.append(np.mean(time))

    print("cnn32 finished")
    return cnn32_time_vs_n


def time_cnn32_2l(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR10(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR10(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_2l_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32_2l = SimpleCNN32Filter2Layers(len({}))
train_loader, test_loader = create_loaders_set(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image(
    cnn32_2l,
    train_loader,
    test_loader,
)
""".format(
                classes, classes, samples
            )
            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            cnn32_2l_time_vs_n.append(np.mean(time))

    print("cnn32_2l finished")
    return cnn32_2l_time_vs_n


def time_cnn32_5l(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR10(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR10(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_5l_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32_5l = SimpleCNN32Filter5Layers(len({}))
train_loader, test_loader = create_loaders_set(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image(
    cnn32_5l,
    train_loader,
    test_loader,
)
""".format(
                classes, classes, samples
            )
            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            cnn32_5l_time_vs_n.append(np.mean(time))

    print("cnn32_5l finished")
    return cnn32_5l_time_vs_n


def time_resnet18(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
cifar_trainset = datasets.CIFAR10(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR10(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    resnet18_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
res = models.resnet18(pretrained=True)
num_ftrs = res.fc.in_features
res.fc = nn.Linear(num_ftrs, len({}))
train_loader, test_loader = create_loaders_set(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image(
    res,
    train_loader,
    test_loader,
)
""".format(
                classes, classes, samples
            )
            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            resnet18_time_vs_n.append(np.mean(time))

    print("resnet18 finished")
    return resnet18_time_vs_n


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    SETUP_CODE = """
from toolbox import (
    run_dn_image,
    create_loaders_set,
    SimpleCNN32Filter,
    SimpleCNN32Filter2Layers,
    SimpleCNN32Filter5Layers,
)
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
"""

    # Parse the class number
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)

    # Run the timers and save results
    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))
    samples_space = np.geomspace(10, 10000, num=8, dtype=int)
    prefix = args.m + "_class/"
    write_result(
        prefix + "cnn32_time_stc.txt",
        time_cnn32(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "cnn32_2l_time_stc.txt",
        time_cnn32_2l(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "cnn32_5l_time_stc.txt",
        time_cnn32_5l(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "resnet18_time_stc.txt",
        time_resnet18(SETUP_CODE, classes_space, samples_space),
    )
