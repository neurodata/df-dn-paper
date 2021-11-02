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
    cnn32_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR100(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR100(
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
            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32,
                train_loader,
                valid_loader,
                test_loader,
            )
            cnn32_kappa.append(cohen_kappa)
            cnn32_ece.append(ece)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result(prefix + "cnn32_kappa.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece.txt", cnn32_ece)
    write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR100(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR100(
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
            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32_2l,
                train_loader,
                valid_loader,
                test_loader,
            )
            cnn32_2l_kappa.append(cohen_kappa)
            cnn32_2l_ece.append(ece)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_kappa.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR100(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR100(
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
            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32_5l,
                train_loader,
                valid_loader,
                test_loader,
            )
            cnn32_5l_kappa.append(cohen_kappa)
            cnn32_5l_ece.append(ece)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_kappa.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            # train data
            cifar_trainset = datasets.CIFAR100(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR100(
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
            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                res,
                train_loader,
                valid_loader,
                test_loader,
            )
            resnet18_kappa.append(cohen_kappa)
            resnet18_ece.append(ece)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result(prefix + "resnet18_kappa.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece.txt", resnet18_ece)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_100.py -m 90
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(100, 10000, num=8, dtype=int)

    nums = list(range(100))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    scale = np.mean(np.arange(0, 256))
    normalize = lambda x: (x - scale) / scale

    # train data
    cifar_trainset = datasets.CIFAR100(
        root="./", train=True, download=True, transform=None
    )
    cifar_train_images = normalize(cifar_trainset.data)
    cifar_train_labels = np.array(cifar_trainset.targets)

    # test data
    cifar_testset = datasets.CIFAR100(
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
