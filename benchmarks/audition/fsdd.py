"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Audrey Zheng
"""
from toolbox import *
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import torchvision.models as models
import warnings
import random

warnings.filterwarnings("ignore")


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
                fsdd_train_images,
                fsdd_train_labels,
                fsdd_test_images,
                fsdd_test_labels,
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
                    fsdd_train_images,
                    fsdd_train_labels,
                    fsdd_valid_images,
                    fsdd_valid_labels,
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
            cnn32 = SimpleCNN32Filter(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
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
            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32_2l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
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
            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                cnn32_5l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
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
            resnet = models.resnet18(pretrained=True)

            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Linear(num_ftrs, len(classes))
            # train data
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            # need to duplicate channel because batch norm cant have 1 channel images
            train_images = torch.cat((train_images, train_images, train_images), dim=1)
            valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)
            test_images = torch.cat((test_images, test_images, test_images), dim=1)

            cohen_kappa, ece, train_time, test_time = run_dn_image_es(
                resnet,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
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
    parser.add_argument("-f", help="feature type")
    args = parser.parse_args()
    n_classes = int(args.m)
    feature_type = str(args.f)
    path_recordings = "recordings/"

    # data is normalized upon loading
    # load dataset
    x_spec, y_number = load_spoken_digit(path_recordings, feature_type)
    nums = list(range(10))
    samples_space = np.geomspace(10, 480, num=6, dtype=int)
    # define path, samples space and number of class combinations
    if feature_type == "melspectrogram":
        prefix = args.m + "_class_mel/"
    elif feature_type == "spectrogram":
        prefix = args.m + "_class/"
    elif feature_type == "mfcc":
        prefix = args.m + "_class_mfcc/"

    # create list of classes with const random seed
    random.Random(5).shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # scale the data
    x_spec = scale(x_spec.reshape(3000, -1), axis=1).reshape(3000, 32, 32)
    y_number = np.array(y_number)

    # need to take train/valid/test equally from each class
    trainx = np.zeros((1, 32, 32))
    trainy = np.zeros((1))
    testx = np.zeros((1, 32, 32))
    testy = np.zeros((1))
    for i in range(10):
        shuffler = np.random.permutation(300)
        trainx = np.concatenate(
            (trainx, x_spec[i * 300 : (i + 1) * 3000][shuffler][:240])
        )
        trainy = np.concatenate(
            (trainy, y_number[i * 300 : (i + 1) * 3000][shuffler][:240])
        )
        testx = np.concatenate(
            (testx, x_spec[i * 300 : (i + 1) * 3000][shuffler][240:])
        )
        testy = np.concatenate(
            (testy, y_number[i * 300 : (i + 1) * 3000][shuffler][240:])
        )
    trainx = trainx[1:]
    trainy = trainy[1:]
    testx = testx[1:]
    testy = testy[1:]

    trainx, validx, trainy, validy = train_test_split(trainx, trainy, test_size=0.2)

    # 3000 samples, 60% train is 1800 samples, 20% test, 20% validation
    fsdd_train_images = trainx.reshape(-1, 32 * 32)
    fsdd_train_labels = trainy.copy()
    # reshape in 2d array
    fsdd_test_images = testx.reshape(-1, 32 * 32)
    fsdd_test_labels = testy.copy()
    fsdd_valid_images = validx.reshape(-1, 32 * 32)
    fsdd_valid_labels = validy.copy()

    # tuning + find the best parameters for rf
    rf_chosen_params_dict = tune_naive_rf()
    run_naive_rf()
    
    #concat valid and test dataset for dn dataset loader's parsing purposes
    testx = np.concatenate((testx, validx))
    fsdd_test_images = testx.reshape(-1, 32 * 32)
    fsdd_test_labels = np.concatenate((fsdd_test_labels, fsdd_valid_labels))
    
    run_cnn32()
    run_cnn32_2l()
    run_cnn32_5l()
    run_resnet18()
