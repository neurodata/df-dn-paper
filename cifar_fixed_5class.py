# general imports
import numpy as np
from toolbox import *
import sklearn
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#%matplotlib inline
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 5))[:45]

# filter python warnings
def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
                
# prepare CIFAR data

# normalize
scale = np.mean(np.arange(0, 256))
normalize = lambda x: (x - scale) / scale

# train data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar_train_images = normalize(cifar_trainset.data)
cifar_train_images = cifar_train_images.reshape(-1, 3072)
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_images = cifar_test_images.reshape(-1, 3072)
cifar_test_labels = np.array(cifar_testset.targets)


# transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  

def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

for classes in experiments:
    fraction_of_train_samples_space = np.geomspace(.001, 1, num=8)
    trials = 1
    num_classes = len(classes)
    
    #resnet18
    resnet_acc = list()
    simplecnn = list() 
    cnn2layer = list()
    complexcnn = list() 
    
    for fraction_of_train_samples in fraction_of_train_samples_space:
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
        train_loader, test_loader = create_loaders_set(cifar_train_labels, cifar_test_labels, classes, trainset, testset, int(fraction_of_train_samples * 10000))
        
        resnet18, input_size = initialize_model('resnet', num_classes, use_pretrained=True)
        best_accuracy = np.mean([run_dn_image(resnet18, train_loader, test_loader) for _ in range(trials)])
        resnet_acc.append(best_accuracy)
        print("resnet Train Fraction:", str(fraction_of_train_samples), "Accuracy:", str(best_accuracy))

        
        cnn32 = SimpleCNN32Filter(num_classes)
        mean_accuracy = np.mean([run_dn_image(cnn32, train_loader, test_loader) for _ in range(trials)])
        simplecnn.append(mean_accuracy)
        print("simple Train Fraction:", str(fraction_of_train_samples), "Accuracy: ", str(mean_accuracy))

        
        cnn32 = SimpleCNN32Filter2Layers(num_classes)
        mean_accuracy = np.mean([run_dn_image(cnn32, train_loader, test_loader) for _ in range(trials)])
        cnn2layer.append(mean_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples), " Cnn 2 layer Accuracy: ", str(mean_accuracy))

        
        cnn32 = CNN5Layer(num_classes)
        mean_accuracy = np.mean([run_dn_image(cnn32, train_loader, test_loader) for _ in range(trials)])
        complexcnn.append(mean_accuracy)
        print("complex Train Fraction:", str(fraction_of_train_samples), " Accuracy: ", str(mean_accuracy))
        
    #naive RF
    rf_acc = list()
    for fraction_of_train_samples in fraction_of_train_samples_space:
        RF = RandomForestClassifier(n_estimators=100, n_jobs = -1)
        best_accuracy = np.mean([run_rf_image_set(RF, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, int(fraction_of_train_samples * 10000), classes) for _ in range(trials)])
        rf_acc.append(best_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
               
    
        
    plt.rcParams['figure.figsize'] = 13, 10
    plt.rcParams['font.size'] = 25
    plt.rcParams['legend.fontsize'] = 16.5
    plt.rcParams['legend.handlelength'] = 2.5
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    

    fig, ax = plt.subplots() # create a new figure with a default 111 subplot
    ax.plot(fraction_of_train_samples_space*10000, rf_acc, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, linestyle=":", label="RF")
    ax.plot(fraction_of_train_samples_space*10000, resnet_acc, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, linestyle="--", label="Resnet18")
    ax.plot(fraction_of_train_samples_space*10000, simplecnn, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, label="simpleCNN")
    ax.plot(fraction_of_train_samples_space*10000, cnn2layer, marker='X', markerfacecolor='red', markersize=8, color='orange', linewidth=3, linestyle=":", label="2layerCNN")
    ax.plot(fraction_of_train_samples_space*10000, complexcnn, marker='X', markerfacecolor='red', markersize=8, color='orange', linewidth=3, label="5layerCNN")


    ax.set_xlabel('Number of Train Samples', fontsize=18)
    ax.set_xscale('log')
    ax.set_xticks([i*10000 for i in list(fraction_of_train_samples_space)])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax.set_ylabel('Accuracy', fontsize=18)
    
    graph_title = str(classes[0]) + " (" + names[classes[0]] + ") "
    file_title = str(classes[0])
    for j in range(1, len(classes)):
        graph_title = graph_title + " vs " + str(classes[j]) + names[classes[j]]
        file_title = file_title + "-" + str(classes[j])
    ax.set_title(graph_title, fontsize=18)
    plt.legend()
    plt.savefig("cifar_results_fixed/" + file_title)
    table = pd.DataFrame(np.concatenate(([rf_acc], [resnet_acc], [simplecnn], [cnn2layer], [complexcnn]), axis=0))
    algos = ['RF', 'resnet', 'simpleCNN', '2layercnn', '5layercnn']
    table['algos'] = algos
    cols = table.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    cols = pd.Index(cols)
    table = table[cols]
    table.to_csv("cifar_results_fixed/" + file_title + ".csv", index=False)
