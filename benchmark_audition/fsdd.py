"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
"""
from audio_toolbox import *
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
import torch
import torch.nn as nn

import torchvision.models as models
import warnings
import random

warnings.filterwarnings("ignore")


def write_result(filename, acc_ls):
    with open(filename, "a") as testwritefile:
        for acc in acc_ls:
            testwritefile.write(str(acc) + "\n")


# prepare FSDD data
def main():
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
    comb = 45
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
    classes_space = list(combinations_45(nums, n_classes, comb))
    print(classes_space)

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

    # 3000 samples, 80% train is 2400 samples, 20% test
    fsdd_train_images = trainx.reshape(-1, 32 * 32)
    fsdd_train_labels = trainy.copy()
    # reshape in 2d array
    fsdd_test_images = testx.reshape(-1, 32 * 32)
    fsdd_test_labels = testy.copy()

    # Resnet18
    resnet18_acc_vs_n = list()
    resnet18_train_time = list()
    resnet18_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (resnet18)
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

            accuracy, train_time, test_time = run_dn_image_es(
                resnet,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            )
            resnet18_acc_vs_n.append(accuracy)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result(prefix + "resnet18.txt", resnet18_acc_vs_n)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)

    # Support Vector Machine
    svm_acc_vs_n = list()
    svm_train_time = list()
    svm_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (svm)
        for samples in samples_space:
            SVM = SVC()
            accuracy, train_time, test_time = run_rf_image_set(
                SVM,
                fsdd_train_images,
                fsdd_train_labels,
                fsdd_test_images,
                fsdd_test_labels,
                samples,
                classes,
            )
            svm_acc_vs_n.append(accuracy)
            svm_train_time.append(train_time)
            svm_test_time.append(test_time)

    print("svm finished")
    write_result(prefix + "svm.txt", svm_acc_vs_n)
    write_result(prefix + "svm_train_time.txt", svm_train_time)
    write_result(prefix + "svm_test_time.txt", svm_test_time)

    # CNN 32
    cnn32_acc_vs_n = list()
    cnn32_train_time = list()
    cnn32_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
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

            accuracy, train_time, test_time = run_dn_image_es(
                cnn32,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            )
            cnn32_acc_vs_n.append(accuracy)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result(prefix + "cnn32.txt", cnn32_acc_vs_n)
    write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)

    # Naive Random Forest
    naive_rf_acc_vs_n = list()
    naive_rf_train_time = list()
    naive_rf_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (naive_rf)
        for samples in samples_space:
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            accuracy, train_time, test_time = run_rf_image_set(
                RF,
                fsdd_train_images,
                fsdd_train_labels,
                fsdd_test_images,
                fsdd_test_labels,
                samples,
                classes,
            )

            naive_rf_acc_vs_n.append(accuracy)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

    print("naive_rf finished")
    write_result(prefix + "naive_rf.txt", naive_rf_acc_vs_n)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)

    # CNN 32 with 2 layers
    cnn32_2l_acc_vs_n = list()
    cnn32_2l_train_time = list()
    cnn32_2l_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        for samples in samples_space:

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

            accuracy, train_time, test_time = run_dn_image_es(
                cnn32_2l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            )
            cnn32_2l_acc_vs_n.append(accuracy)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l.txt", cnn32_2l_acc_vs_n)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)

    # CNN 32 with 5 layers
    cnn32_5l_acc_vs_n = list()
    cnn32_5l_train_time = list()
    cnn32_5l_test_time = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        for samples in samples_space:

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

            accuracy, train_time, test_time = run_dn_image_es(
                cnn32_5l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            )

            cnn32_5l_acc_vs_n.append(accuracy)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l.txt", cnn32_5l_acc_vs_n)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
