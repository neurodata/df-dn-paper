# general imports
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")


plt.rcParams["legend.loc"] = "best"
plt.rcParams['figure.facecolor'] = 'white'
#%matplotlib inline
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
cifar_train_labels = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
cifar_test_images = normalize(cifar_testset.data)
cifar_test_labels = np.array(cifar_testset.targets)


# transform
data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=data_transforms)
testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=data_transforms)


    
def run_rf(model, train_images, train_labels, test_images, test_labels, fraction_of_train_samples, class1=0, class2=2):
    
    class1_indices = np.argwhere(train_labels==class1).flatten()
    np.random.shuffle(class1_indices)
    class1_indices = class1_indices[:int(len(class1_indices) * fraction_of_train_samples)]
    
    class2_indices = np.argwhere(train_labels==class2).flatten()
    np.random.shuffle(class2_indices)
    class2_indices = class2_indices[:int(len(class2_indices) * fraction_of_train_samples)]

    train_indices = np.concatenate([class1_indices, class2_indices]) 
    np.random.shuffle(train_indices)
    # get only train images and labels for class 1 and class 2
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    # get only test images and labels for class 1 and class 2
    class1_indices = np.argwhere(test_labels==class1).flatten()
    class2_indices = np.argwhere(test_labels==class2).flatten()
    test_indices = np.concatenate([class1_indices, class2_indices]) 
    test_images = test_images[test_indices]

    train_images = train_images.reshape(-1, 32*32*3)
    test_images = test_images.reshape(-1, 32*32*3)
    
    model.fit(train_images, train_labels)
    # Test
    test_preds = model.predict(test_images)
    return accuracy_score(test_labels[test_indices], test_preds)


def run_res(net, train_labels, test_labels, fraction_of_train_samples, trainset, testset, class1=0, class2=2, epochs=5, lr=.001, batch=16):

    class1_indices = np.argwhere(train_labels==class1).flatten()
    np.random.shuffle(class1_indices)
    class1_indices = class1_indices[:int(len(class1_indices) * fraction_of_train_samples)]
    for i in range(len(trainset.targets)):
        if i in class1_indices:
            trainset.targets[i] = 0
    
    class2_indices = np.argwhere(train_labels==class2).flatten()
    np.random.shuffle(class2_indices)
    class2_indices = class2_indices[:int(len(class2_indices) * fraction_of_train_samples)]
    for i in range(len(trainset.targets)):
        if i in class2_indices:
            trainset.targets[i] = 1
            
    train_indices = np.concatenate([class1_indices, class2_indices]) 
    np.random.shuffle(train_indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, num_workers=4, sampler=train_sampler)


    class1_indices = np.argwhere(test_labels==class1).flatten()
    class2_indices = np.argwhere(test_labels==class2).flatten()
    for i in range(len(testset.targets)):
        if i in class1_indices:
            testset.targets[i] = 0
    
    for i in range(len(testset.targets)):
        if i in class2_indices:
            testset.targets[i] = 1
    test_indices = np.concatenate([class1_indices, class2_indices])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False, num_workers=4, sampler=test_sampler)

    # define model
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = torch.tensor(inputs).to(dev)
            labels = torch.tensor(labels).to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # test the model
    correct = torch.tensor(0).to(dev)
    total = torch.tensor(0).to(dev)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = torch.tensor(labels).to(dev)
            images = torch.tensor(images).to(dev)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    accuracy = float(correct) / float(total)
    return accuracy

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

exit()

for class1 in range(1):
    for class2 in range(class1 + 1, 12):
        fraction_of_train_samples_space = np.geomspace(1, 1, num=1)
        
        
        #naive RF
        rf_acc = list()
        for fraction_of_train_samples in fraction_of_train_samples_space:
            RF = RandomForestClassifier(n_estimators=100, n_jobs = -1)
            best_accuracy = np.mean([run_rf(RF, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples) for _ in range(1)])
            rf_acc.append(best_accuracy)
            print("Train Fraction:", str(fraction_of_train_samples))
            print("Accuracy:", str(best_accuracy))
            
        #resnet18
        resnet_acc = list()
        for fraction_of_train_samples in fraction_of_train_samples_space:
            resnet18, input_size = initialize_model('resnet', 2, use_pretrained=True)
            best_accuracy = np.mean([run_res(resnet18, cifar_train_labels, cifar_test_labels, fraction_of_train_samples, trainset, testset) for _ in range(1)])
            resnet_acc.append(best_accuracy)
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
       
        ax.set_xlabel('Number of Train Samples', fontsize=18)
        ax.set_xscale('log')
        ax.set_xticks([i*10000 for i in list(np.geomspace(0.01, 1, num=8))])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        ax.set_ylabel('Accuracy', fontsize=18)
        
        ax.set_title(str(class1) + " (" + names[class1] + ") vs " + str(class2) + "(" + names[class2] + ") classification", fontsize=18)
        plt.legend()
        plt.savefig("cifar_results/" + str(class1) + "_vs_" + str(class2))
        table = pd.DataFrame(np.concatenate(([rf_acc], [resnet_acc]), axis=0))
        algos = ['RF', 'resnet']
        table['algos'] = algos
        cols = table.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        cols = pd.Index(cols)
        table = table[cols]
        table.to_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index=False)
