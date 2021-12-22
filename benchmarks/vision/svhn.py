"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Audrey Zheng
"""
from svhn_toolbox import *
from toolbox import *

import argparse
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
            clf = sklearn.ensemble.RandomForestClassifier()
            clf.set_params(**rf_chosen_params_dict[(classes, samples)])
            cohen_kappa, ece, train_time, test_time = run_rf_image_set(
                clf,
                svhn_train_images,
                svhn_train_labels,
                svhn_test_images,
                svhn_test_labels,
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


def tune_naive_rf():
    # only tuning on accuracy, regardless of training time
    naive_rf_accuracy = []
    naive_rf_train_time = []
    naive_rf_val_time = []
    naive_rf_param_dict = {}
    
    for classes in classes_space:
        
        for samples in samples_space:
            # specify how many random combinations we are going to test out
            num_iter = 6
            candidate_param_list = parameter_list_generator(num_iter)
            
            for params in candidate_param_list:
                clf = sklearn.ensemble.RandomForestClassifier()
                clf.set_params(**params)
                accuracy, train_time, val_time = tune_rf_image_set(
                    clf,
                    svhn_train_images,
                    svhn_train_labels,
                    svhn_val_images,
                    svhn_val_labels,
                    samples,
                    classes,
                )
                naive_rf_accuracy.append(accuracy)
                naive_rf_train_time.append(train_time)
                naive_rf_val_time.append(val_time)
            
            max_accuracy = max(naive_rf_accuracy)
            max_index = naive_rf_accuracy.index(max_accuracy)
            naive_rf_param_dict[(classes,samples)] = candidate_param_list[max_index]

    print("naive_rf tuning finished")
    write_result(prefix + "naive_rf_parameters_dict.txt", naive_rf_param_dict)
    write_result(prefix + "naive_rf_parameters_tuning_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_parameters_tuning_val_time.txt", naive_rf_val_time)
    write_result(prefix + "naive_rf_parameters_tuning_accuracy.txt", naive_rf_accuracy)
    
    return naive_rf_param_dict


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (cnn32)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/svhn_"
    samples_space = np.geomspace(10, 10000, num=8, dtype=int)

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    scale = np.mean(np.arange(0, 256))
    normalize = lambda x: (x - scale) / scale

    """
    Originally, the SVHN dataset had a train:test split of 73257:26032 digits.
    I will extract the test and training data below, contactenate them into a big set, and then split them into a 2:1:1 train:test:validate split.
    """
    # original train data 
    svhn_trainset = datasets.SVHN(
        root="./", split="train", download=True, transform=None
    )
    svhn_train_images = normalize(svhn_trainset.data)
    svhn_train_labels = np.array(svhn_trainset.targets)
    
    # original test data
    svhn_testset = datasets.SVHN(
    root="./", split="test", download=True, transform=None
    )
    svhn_test_images = normalize(svhn_testset.data)
    svhn_test_labels = np.array(svhn_testset.targets)
    
    # train data concatenation (train: 100%)
    svhn_train_images = np.concatenate((svhn_train_images, svhn_test_images))
    svhn_train_labels = np.concatenate((svhn_train_labels, svhn_test_labels))
    
    # train data and validation data initial split (train: 50%; val: 50%)
    svhn_train_images, svhn_val_images, svhn_train_labels, svhn_val_labels = train_test_split(svhn_train_images, svhn_train_labels, shuffle=True, test_size=0.5)
    
    # validation data further split (train: 50%; val: 25%; test: 25%)
    svhn_val_images, svhn_test_images, svhn_val_labels, svhn_test_labels = train_test_split(svhn_val_images, svhn_val_labels, shuffle=True, test_size=0.5)

    svhn_train_images = svhn_train_images.reshape(-1, 32 * 32 * 3)
    svhn_val_images = svhn_val_images.reshape(-1, 32 * 32 * 3)
    svhn_test_images = svhn_test_images.reshape(-1, 32 * 32 * 3)
    
    # tuning + find the best parameters
    rf_chosen_params_dict = tune_naive_rf()

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
