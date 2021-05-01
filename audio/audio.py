"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
"""
from audio_toolbox import *

import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from spoken_digit_functions import *
import warnings

warnings.filterwarnings('ignore')

def write_result(filename, acc_ls):
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")

def combinations_45(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    count = 0
    while count < 2: #TODO CHANGE TO 45!!!!!!!!!!!!!!!
        count += 1
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# prepare CIFAR data
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"

    nums = range(10)
    classes_space = list(combinations_45(nums, n_classes))

    path_recordings = 'recordings/'
    #data is normalized upon loading
    x_spec, y_number = load_spoken_digit(path_recordings)
    y_number = np.array(y_number)
    
    #need to take train/valid/test equally from each class
    trainx = np.zeros((1, 32, 32))
    trainy = np.zeros((1))
    testx = np.zeros((1, 32, 32))
    testy = np.zeros((1))
    for i in range(10):
        shuffler = np.random.permutation(300)
        trainx = np.concatenate((trainx, x_spec[i * 300: (i + 1) * 3000][shuffler][:240]))
        trainy = np.concatenate((trainy, y_number[i * 300: (i + 1) * 3000][shuffler][:240]))
        testx = np.concatenate((testx, x_spec[i * 300: (i + 1) * 3000][shuffler][240:]))
        testy = np.concatenate((testy, y_number[i * 300: (i + 1) * 3000][shuffler][240:])) 
    trainx = trainx[1:]
    trainy = trainy[1:]           
    testx = testx[1:]
    testy = testy[1:]
    samples_space = np.geomspace(10, 240, num=6, dtype=int)
    
    #3000 samples, 80% train is 2400 samples, 20% test
    cifar_train_images = trainx.reshape(-1, 32 * 32)
    cifar_train_labels = trainy.copy()
    #reshape in 2d array
    cifar_test_images = testx.reshape(-1, 32 * 32)
    cifar_test_labels = testy.copy()




    resnet18_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (resnet18)
        for samples in samples_space:
            #print(samples)
            model = models.resnet18(pretrained=True)
            #model.conv1 = nn.Conv2d(2, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(classes))
            # train data
            #3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            #reshape in 4d array
            test_images =  testx.copy()
            test_labels =  testy.copy()

            train_images, train_labels, valid_images, valid_labels, test_images, \
                test_labels = prepare_data(train_images, train_labels, test_images, \
                                           test_labels, samples, classes)
            #print(train_images.shape)
            #need to duplicate channel because batch norm cant have 1 channel images
            train_images = torch.cat((train_images, train_images, train_images), dim=1)
            valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)
            test_images = torch.cat((test_images, test_images, test_images), dim=1)
            #print(train_images.shape)
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        model,
                        train_images, train_labels,
                        valid_images, valid_labels,
                        test_images, test_labels,
                    )
                    for _ in range(1)
                ]
            )
            resnet18_acc_vs_n.append(mean_accuracy)

    print("resnet18 finished")
    write_result(prefix + "resnet18.txt", resnet18_acc_vs_n)
    

    
    svm_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (svm)
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
    write_result(prefix + "svm.txt", svm_acc_vs_n)



    cnn32_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32)
        for samples in samples_space:
            # train data
            model = SimpleCNN32Filter(len(classes))
            #3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            #reshape in 4d array
            test_images =  testx.copy()
            test_labels =  testy.copy()

            train_images, train_labels, valid_images, valid_labels, test_images, \
                test_labels = prepare_data(train_images, train_labels, test_images, \
                                           test_labels, samples, classes)
            
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        model,
                        train_images, train_labels,
                        valid_images, valid_labels,
                        test_images, test_labels,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_acc_vs_n.append(mean_accuracy)

    print("cnn32 finished")
    write_result(prefix + "cnn32.txt", cnn32_acc_vs_n)
    
    
    
    naive_rf_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (naive_rf)
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
    write_result(prefix + "naive_rf.txt", naive_rf_acc_vs_n)
    


    cnn32_2l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_2l)
        for samples in samples_space:
            
            model = SimpleCNN32Filter2Layers(len(classes))
            #3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            #reshape in 4d array
            test_images =  testx.copy()
            test_labels =  testy.copy()

            train_images, train_labels, valid_images, valid_labels, test_images, \
                test_labels = prepare_data(train_images, train_labels, test_images, \
                                           test_labels, samples, classes)
            
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        model,
                        train_images, train_labels,
                        valid_images, valid_labels,
                        test_images, test_labels,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_2l_acc_vs_n.append(mean_accuracy)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l.txt", cnn32_2l_acc_vs_n)



    cnn32_5l_acc_vs_n = list()
    for classes in classes_space:

        # accuracy vs num training samples (cnn32_5l)
        for samples in samples_space:
            
            model = SimpleCNN32Filter5Layers(len(classes))
            #3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            #reshape in 4d array
            test_images =  testx.copy()
            test_labels =  testy.copy()

            train_images, train_labels, valid_images, valid_labels, test_images, \
                test_labels = prepare_data(train_images, train_labels, test_images, \
                                           test_labels, samples, classes)
            
            mean_accuracy = np.mean(
                [
                    run_dn_image_es(
                        model,
                        train_images, train_labels,
                        valid_images, valid_labels,
                        test_images, test_labels,
                    )
                    for _ in range(1)
                ]
            )
            cnn32_5l_acc_vs_n.append(mean_accuracy)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l.txt", cnn32_5l_acc_vs_n)



if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()