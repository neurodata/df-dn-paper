"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
"""
from svnh_toolbox import *

import argparse
import random
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


# prepare CIFAR data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    scale = np.mean(np.arange(0, 256))
    normalize = lambda x: (x - scale) / scale

    # train data
    svhn_trainset = datasets.SVHN(
        root="./", split="train", download=True, transform=None
    )
    svhn_train_images = normalize(svhn_trainset.data)
    svhn_train_labels = np.array(svhn_trainset.labels)

    # test data
    svhn_testset = datasets.SVHN(root="./", split="test", download=True, transform=None)
    svhn_test_images = normalize(svhn_testset.data)
    svhn_test_labels = np.array(svhn_testset.labels)

    svhn_train_images = svhn_train_images.reshape(-1, 32 * 32 * 3)
    svhn_test_images = svhn_test_images.reshape(-1, 32 * 32 * 3)

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
                        svhn_train_images,
                        svhn_train_labels,
                        svhn_test_images,
                        svhn_test_labels,
                        samples,
                        classes,
                    )
                    for _ in range(1)
                ]
            )
            svm_acc_vs_n.append(mean_accuracy)

    print("svm finished")
    write_result(prefix + "svnh_svm.txt", svm_acc_vs_n)

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
                        svhn_train_images,
                        svhn_train_labels,
                        svhn_test_images,
                        svhn_test_labels,
                        samples,
                        classes,
                    )
                    for _ in range(1)
                ]
            )
            naive_rf_acc_vs_n.append(mean_accuracy)

    print("naive_rf finished")
    write_result(prefix + "svnh_naive_rf.txt", naive_rf_acc_vs_n)

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cnn32_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for samples in samples_space:
            # train data
            svhn_trainset = datasets.SVHN(
                root="./", split="train", download=True, transform=data_transforms
            )
            svhn_train_labels = np.array(svhn_trainset.labels)

            # test data
            svhn_testset = datasets.SVHN(
                root="./", split="test", download=True, transform=data_transforms
            )
            svhn_test_labels = np.array(svhn_testset.labels)

            cnn32 = SimpleCNN32Filter(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                svhn_train_labels,
                svhn_test_labels,
                classes,
                svhn_trainset,
                svhn_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        cnn32,
                        train_loader,
                        valid_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_acc_vs_n.append(mean_accuracy)

    print("cnn32 finished")
    write_result(prefix + "svnh_cnn32.txt", cnn32_acc_vs_n)

    cnn32_2l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for samples in samples_space:
            # train data
            svhn_trainset = datasets.SVHN(
                root="./", split="train", download=True, transform=data_transforms
            )
            svhn_train_labels = np.array(svhn_trainset.labels)

            # test data
            svhn_testset = datasets.SVHN(
                root="./", split="test", download=True, transform=data_transforms
            )
            svhn_test_labels = np.array(svhn_testset.labels)

            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                svhn_train_labels,
                svhn_test_labels,
                classes,
                svhn_trainset,
                svhn_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        cnn32_2l,
                        train_loader,
                        valid_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_2l_acc_vs_n.append(mean_accuracy)

    print("cnn32_2l finished")
    write_result(prefix + "svnh_cnn32_2l.txt", cnn32_2l_acc_vs_n)

    cnn32_5l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        samples_space = np.geomspace(10, 10000, num=8, dtype=int)
        for samples in samples_space:
            # train data
            svhn_trainset = datasets.SVHN(
                root="./", split="train", download=True, transform=data_transforms
            )
            svhn_train_labels = np.array(svhn_trainset.labels)

            # test data
            svhn_testset = datasets.SVHN(
                root="./", split="test", download=True, transform=data_transforms
            )
            svhn_test_labels = np.array(svhn_testset.labels)

            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                svhn_train_labels,
                svhn_test_labels,
                classes,
                svhn_trainset,
                svhn_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        cnn32_5l,
                        train_loader,
                        valid_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_5l_acc_vs_n.append(mean_accuracy)

    print("cnn32_5l finished")
    write_result(prefix + "svnh_cnn32_5l.txt", cnn32_5l_acc_vs_n)

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
            svhn_trainset = datasets.SVHN(
                root="./", split="train", download=True, transform=data_transforms
            )
            svhn_train_labels = np.array(svhn_trainset.labels)

            # test data
            svhn_testset = datasets.SVHN(
                root="./", split="test", download=True, transform=data_transforms
            )
            svhn_test_labels = np.array(svhn_testset.labels)

            res = models.resnet18(pretrained=True)
            num_ftrs = res.fc.in_features
            res.fc = nn.Linear(num_ftrs, len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                svhn_train_labels,
                svhn_test_labels,
                classes,
                svhn_trainset,
                svhn_testset,
                samples,
            )
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        res,
                        train_loader,
                        valid_loader,
                        test_loader,
                    )
                    for _ in range(1)
                ]
            )
            resnet18_acc_vs_n.append(mean_accuracy)

    print("resnet18 finished")
    write_result(prefix + "svnh_resnet18.txt", resnet18_acc_vs_n)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
