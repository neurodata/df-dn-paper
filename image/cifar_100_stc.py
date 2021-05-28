from toolbox import *

import argparse
import random

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# prepare CIFAR data
def main():
    # Example usage: python cifar_100.py -m 90 -s l
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-s", help="computation speed")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"

    nums = list(range(100))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    if args.s == "h":
        # High speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_train_time.txt"))
        suffix = "_st.txt"
        ratio = 1.0
    elif args.s == "l":
        # Low speed RF
        rf_times = produce_mean(load_result(prefix + "naive_rf_train_time_lc.txt"))
        suffix = "_sc.txt"
        ratio = 0.11 / 0.9
    else:
        raise Exception("Wrong configurations for time calibration.")

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cnn32_acc_vs_n = list()
    cnn32_train_time = list()
    cnn32_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
        samples_space = np.geomspace(100, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            accuracy, train_time, test_time = run_dn_image_set(
                cnn32,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_acc_vs_n.append(accuracy)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result(prefix + "cnn32" + suffix, cnn32_acc_vs_n)
    write_result(prefix + "cnn32_train_time" + suffix, cnn32_train_time)
    write_result(prefix + "cnn32_test_time" + suffix, cnn32_test_time)

    cnn32_2l_acc_vs_n = list()
    cnn32_2l_train_time = list()
    cnn32_2l_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        samples_space = np.geomspace(100, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            accuracy, train_time, test_time = run_dn_image_set(
                cnn32_2l,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_2l_acc_vs_n.append(accuracy)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l" + suffix, cnn32_2l_acc_vs_n)
    write_result(prefix + "cnn32_2l_train_time" + suffix, cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time" + suffix, cnn32_2l_test_time)

    cnn32_5l_acc_vs_n = list()
    cnn32_5l_train_time = list()
    cnn32_5l_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        samples_space = np.geomspace(100, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            accuracy, train_time, test_time = run_dn_image_set(
                cnn32_5l,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            cnn32_5l_acc_vs_n.append(accuracy)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l" + suffix, cnn32_5l_acc_vs_n)
    write_result(prefix + "cnn32_5l_train_time" + suffix, cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time" + suffix, cnn32_5l_test_time)

    # prepare CIFAR data
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    resnet18_acc_vs_n = list()
    resnet18_train_time = list()
    resnet18_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (resnet18)
        samples_space = np.geomspace(100, 10000, num=8, dtype=int)
        for i, samples in enumerate(samples_space):
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
            time_limit = rf_times[i]
            train_loader, test_loader = create_loaders_set(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
            )
            accuracy, train_time, test_time = run_dn_image_set(
                res,
                train_loader,
                test_loader,
                time_limit=time_limit,
                ratio=ratio,
            )
            resnet18_acc_vs_n.append(accuracy)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result(prefix + "resnet18" + suffix, resnet18_acc_vs_n)
    write_result(prefix + "resnet18_train_time" + suffix, resnet18_train_time)
    write_result(prefix + "resnet18_test_time" + suffix, resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
