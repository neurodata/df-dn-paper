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


def time_svm(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=None
)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
    root="./", train=False, download=True, transform=None
)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)"""
    )

    svm_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
SVM = SVC()
run_rf_image_set(
    SVM,
    cifar_train_images,
    cifar_train_labels,
    cifar_test_images,
    cifar_test_labels,
    {},
    {},
)""".format(
                samples, classes
            )
            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            svm_time_vs_n.append(np.mean(time))

    print("svm finished")
    return svm_time_vs_n


def time_rf(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=None
)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
    root="./", train=False, download=True, transform=None
)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)

cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)"""
    )

    naive_rf_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
run_rf_image_set(
    RF,
    cifar_train_images,
    cifar_train_labels,
    cifar_test_images,
    cifar_test_labels,
    {},
    {},
)""".format(
                samples, classes
            )

            time = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=1, number=1)
            naive_rf_time_vs_n.append(np.mean(time))

    print("naive_rf finished")
    return naive_rf_time_vs_n


def time_cnn32(SETUP_CODE, classes_space, samples_space):
    SETUP_CODE = (
        SETUP_CODE
        + """
data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32 = SimpleCNN32Filter(len({}))
train_loader, valid_loader, test_loader = create_loaders_es(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image_es(
    cnn32,
    train_loader,
    valid_loader,
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
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_2l_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32_2l = SimpleCNN32Filter2Layers(len({}))
train_loader, valid_loader, test_loader = create_loaders_es(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image_es(
    cnn32_2l,
    train_loader,
    valid_loader,
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
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
    root="./", train=False, download=True, transform=data_transforms
)
cifar_test_labels = np.array(cifar_testset.targets)"""
    )

    cnn32_5l_time_vs_n = list()
    for classes in classes_space:
        for samples in samples_space:
            TEST_CODE = """
cnn32_5l = SimpleCNN32Filter5Layers(len({}))
train_loader, valid_loader, test_loader = create_loaders_es(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image_es(
    cnn32_5l,
    train_loader,
    valid_loader,
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
cifar_trainset = datasets.CIFAR100(
    root="./", train=True, download=True, transform=data_transforms
)
cifar_train_labels = np.array(cifar_trainset.targets)

cifar_testset = datasets.CIFAR100(
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
train_loader, valid_loader, test_loader = create_loaders_es(
    cifar_train_labels,
    cifar_test_labels,
    {},
    cifar_trainset,
    cifar_testset,
    {},
)
run_dn_image_es(
    res,
    train_loader,
    valid_loader,
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
    run_rf_image_set,
    run_dn_image_es,
    create_loaders_es,
    SimpleCNN32Filter,
    SimpleCNN32Filter2Layers,
    SimpleCNN32Filter5Layers,
)
import numpy as np
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

scale = np.mean(np.arange(0, 256))
normalize = lambda x: (x - scale) / scale
"""

    # Parse the class number
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)

    # Run the timers and save results
    nums = list(range(100))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))
    samples_space = np.geomspace(100, 10000, num=8, dtype=int)
    prefix = args.m + "_class/"
    write_result(
        prefix + "naive_rf_time.txt", time_rf(SETUP_CODE, classes_space, samples_space)
    )
    write_result(
        prefix + "cnn32_time.txt", time_cnn32(SETUP_CODE, classes_space, samples_space)
    )
    write_result(
        prefix + "cnn32_2l_time.txt",
        time_cnn32_2l(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "cnn32_5l_time.txt",
        time_cnn32_5l(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "resnet18_time.txt",
        time_resnet18(SETUP_CODE, classes_space, samples_space),
    )
    write_result(
        prefix + "svm_time.txt", time_svm(SETUP_CODE, classes_space, samples_space)
    )
