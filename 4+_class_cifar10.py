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
experiments = [(0, 2, 6, 8), (1, 3, 4, 9), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6)\
               , (0, 1, 2, 7), (0, 1, 2, 8), (0, 1, 2, 9)]
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
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  

    
def run_rf(model, train_images, train_labels, test_images, test_labels, fraction_of_train_samples, classes):
    
    train_indices = []
    #iterate through classes and get their indices
    for i in classes:
        class_indices = np.argwhere(train_labels==i).flatten()
        np.random.shuffle(class_indices)
        class_indices = class_indices[:int(len(class_indices) * fraction_of_train_samples)]
        train_indices = np.concatenate([train_indices, class_indices])
        
        
    train_indices = train_indices.astype(int)
    np.random.shuffle(train_indices)
    # get only train images and labels for class 1 and class 2
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    # get only test images and labels for class 1 and class 2
    test_indices = []
    for i in classes:
        class_indices = np.argwhere(test_labels==i).flatten()
        np.random.shuffle(class_indices)
        class_indices = class_indices[:int(len(class_indices) * fraction_of_train_samples)]
        test_indices = np.concatenate([test_indices, class_indices])
    
    test_indices = test_indices.astype(int)
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

class SimpleCNN32Filter2Layers(torch.nn.Module):
    
    def __init__(self):
        super(SimpleCNN32Filter2Layers, self).__init__()        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(8192, 200)
        self.fc2 = torch.nn.Linear(200, 10)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.maxpool(x)
        x = x.view(b, -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)
    
def run_cnn(cnn_model, cifar_train_labels, cifar_test_labels, fraction_of_train_samples, class1=3, class2=5):
    # set params
    num_epochs = 20
    learning_rate = 0.001

    class1_indices = np.argwhere(cifar_train_labels==class1).flatten()
    class1_indices = class1_indices[:int(len(class1_indices) * fraction_of_train_samples)]
    class2_indices = np.argwhere(cifar_train_labels==class2).flatten()
    class2_indices = class2_indices[:int(len(class2_indices) * fraction_of_train_samples)]
    train_indices = np.concatenate([class1_indices, class2_indices]) 
    
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler)

    test_indices = np.concatenate([np.argwhere(cifar_test_labels==class1).flatten(), np.argwhere(cifar_test_labels==class2).flatten()])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, sampler=test_sampler)

    # define model
    net2 = SimpleCNN32Filter2Layers()
    dev = torch.device("cuda:0")
    net2.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate)
    print("here")
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = torch.tensor(inputs).to(dev)
            labels = torch.tensor(labels).to(dev)
            # zero the parameter gradients
            optimizer2.zero_grad()
            # forward + backward + optimize
            outputs2 = net2(inputs)
            
            loss2 = criterion(outputs2, labels)
            
            loss2.backward()
            
            optimizer2.step()
            
        # test the model
        total = torch.tensor(0).to(dev)
        correct2 = torch.tensor(0).to(dev)
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                labels = torch.tensor(labels).to(dev)
                images = torch.tensor(images).to(dev)
                outputs2 = net2(images)
                _, predicted2 = torch.max(outputs2.data, 1)
                total += labels.size(0)
                correct2 += (predicted2 == labels.view(-1)).sum().item()
        accuracy2 = float(correct2) / float(total)
        print("epoch: ", epoch, "complicated: ", accuracy2)
    return accuracy2


for classes in experiments:
    fraction_of_train_samples_space = np.geomspace(.002, .2, num=3)
    trials = 1
    
    #naive RF
    rf_acc = list()
    for fraction_of_train_samples in fraction_of_train_samples_space:
        RF = RandomForestClassifier(n_estimators=100, n_jobs = -1)
        best_accuracy = np.mean([run_rf(RF, cifar_train_images, cifar_train_labels, cifar_test_images, cifar_test_labels, fraction_of_train_samples, classes) for _ in range(trials)])
        rf_acc.append(best_accuracy)
        print("Train Fraction:", str(fraction_of_train_samples))
        print("Accuracy:", str(best_accuracy))
     
            
        '''
        #resnet18
        resnet_acc = list()
        for fraction_of_train_samples in fraction_of_train_samples_space:
            resnet18, input_size = initialize_model('resnet', 2, use_pretrained=True)
            best_accuracy = np.mean([run_res(resnet18, cifar_train_labels, cifar_test_labels, fraction_of_train_samples, trainset, testset) for _ in range(trials)])
            resnet_acc.append(best_accuracy)
            print("Train Fraction:", str(fraction_of_train_samples))
            print("Accuracy:", str(best_accuracy))
            
        cnn32_two_layer = list()    
        for fraction_of_train_samples in fraction_of_train_samples_space:
             accu_cnn32_2 = 0
             for i in range(trials):
                 tempcnn32_2 = run_cnn(cifar_train_labels, cifar_test_labels, fraction_of_train_samples)
                 accu_cnn32_2 += tempcnn32_2
             cnn32_two_layer.append(accu_cnn32_2 / trials)
             print("Train Fraction:", str(fraction_of_train_samples))
             print(" Cnn 2 layer Accuracy: ", str(accu_cnn32_2 / trials))
          '''         
          
        
        
    plt.rcParams['figure.figsize'] = 13, 10
    plt.rcParams['font.size'] = 25
    plt.rcParams['legend.fontsize'] = 16.5
    plt.rcParams['legend.handlelength'] = 2.5
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    
    fig, ax = plt.subplots() # create a new figure with a default 111 subplot
    ax.plot(fraction_of_train_samples_space*10000, rf_acc, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, linestyle=":", label="RF")
    #ax.plot(fraction_of_train_samples_space*10000, resnet_acc, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, linestyle="--", label="Resnet18")
    #ax.plot(fraction_of_train_samples_space*10000, cnn32_two_layer, marker='X', markerfacecolor='red', markersize=8, color='green', linewidth=3, label="CNN")

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
    plt.savefig("cifar_results_4+/" + file_title)
    #table = pd.DataFrame(np.concatenate(([rf_acc], [resnet_acc], [cnn32_two_layer]), axis=0))
    table = pd.DataFrame([rf_acc])
    #algos = ['RF', 'resnet', 'CNN']
    algos = ['RF']
    table['algos'] = algos
    cols = table.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    cols = pd.Index(cols)
    table = table[cols]
    table.to_csv("cifar_results_4+/" + file_title + ".csv", index=False)
