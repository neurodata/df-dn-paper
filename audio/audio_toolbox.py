"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleCNN32Filter(nn.Module):
    """
    Defines a simple CNN arhcitecture
    """

    def __init__(self, num_classes):
        super(SimpleCNN32Filter, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(144 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144 * 32)
        x = self.fc1(x)
        return x


class SimpleCNN32Filter2Layers(nn.Module):
    """
    Define a simple CNN arhcitecture with 2 layers
    """

    def __init__(self, num_classes):
        super(SimpleCNN32Filter2Layers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(12 * 12 * 32, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN32Filter5Layers(torch.nn.Module):
    """
    Define a simple CNN arhcitecture with 5 layers
    """

    def __init__(self, num_classes):
        super(SimpleCNN32Filter5Layers, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(8192, 200)
        self.fc2 = torch.nn.Linear(200, num_classes)
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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def combinations_45(iterable, r):
    """Extracts 45 combinations from given list"""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    count = 0
    while count < 44:
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


def run_rf_image_set(
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    samples,
    classes,
):
    """
    Peforms multiclass predictions for a random forest classifier
    with fixed total samples
    """
    num_classes = len(classes)
    partitions = np.array_split(np.array(range(samples)), num_classes)

    # Obtain only train images and labels for selected classes
    image_ls = []
    label_ls = []
    i = 0
    for cls in classes:
        class_idx = np.argwhere(train_labels == cls).flatten()
        np.random.shuffle(class_idx)
        class_img = train_images[class_idx[: len(partitions[i])]]
        image_ls.append(class_img)
        label_ls.append(np.repeat(cls, len(partitions[i])))
        i += 1

    train_images = np.concatenate(image_ls)
    train_labels = np.concatenate(label_ls)
    print(train_images.shape, train_labels.shape)
    # Obtain only test images and labels for selected classes
    image_ls = []
    label_ls = []
    for cls in classes:
        image_ls.append(test_images[test_labels == cls])
        label_ls.append(np.repeat(cls, np.sum(test_labels == cls)))

    test_images = np.concatenate(image_ls)
    test_labels = np.concatenate(label_ls)

    # Train the model
    model.fit(train_images, train_labels)

    # Test the model
    test_preds = model.predict(test_images)

    return accuracy_score(test_labels, test_preds)


def run_dn_image_es(
    model,
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    test_data, 
    test_labels,
    epochs=30,
    lr=0.001,
    batch=64,
):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """
    # define model
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # early stopping setup
    prev_loss = float("inf")
    flag = 0

    for epoch in range(epochs):  # loop over the dataset multiple times

        for i in range(len(train_data)):
            # get the inputs
            inputs = train_data[i:i + batch].to(dev)
            labels = train_labels[i:i + batch].to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # test generalization error for early stopping
        cur_loss = 0
        with torch.no_grad():
            for i in range(len(valid_data)):
                # get the inputs
                inputs = valid_data[i:i + batch].to(dev)
                labels = valid_labels[i:i + batch].to(dev)

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                cur_loss += loss
        # early stop if 3 epochs in a row no loss decrease
        if cur_loss < prev_loss:
            prev_loss = cur_loss
            flag = 0
        else:
            flag += 1
            if flag >= 3:
                print("early stopped at epoch: ", epoch)
                break

    # test the model
    correct = torch.tensor(0).to(dev)
    total = torch.tensor(0).to(dev)
    with torch.no_grad():
        for i in range(len(test_data)):
            inputs = test_data[i:i + batch].to(dev)
            labels = test_labels[i:i + batch].to(dev)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    accuracy = float(correct) / float(total)
    return accuracy


def prepare_data(train_images, train_labels, test_images, \
    test_labels, samples, classes):

    classes = np.array(list(classes))
    num_classes = len(classes)
    partitions = np.array_split(np.array(range(samples)), num_classes)
    # get indicies of classes we want
    class_idxs = []
    i = 0
    for cls in classes:
        class_idx = np.argwhere(train_labels == cls).flatten()
        np.random.shuffle(class_idx)
        class_idx = class_idx[: len(partitions[i])]
        class_idxs.append(class_idx)
        i += 1


    train_idxs = np.concatenate(class_idxs)
    np.random.shuffle(train_idxs)
    # change the labels to be from 0-len(classes)
    for i in train_idxs:
        train_labels[i] = np.where(classes == train_labels[i])[0][0]
    
    # get indicies of classes we want
    test_idxs = []
    validation_idxs = []
    for cls in classes:
        test_idx = np.argwhere(test_labels == cls).flatten()
        # out of all, 0.3 validation, 0.7 test
        test_idxs.append(test_idx[int(len(test_idx) * 0.3) :])
        validation_idxs.append(test_idx[: int(len(test_idx) * 0.3)])

    test_idxs = np.concatenate(test_idxs)
    validation_idxs = np.concatenate(validation_idxs)

    # change the labels to be from 0-len(classes)
    for i in test_idxs:
        test_labels[i] = np.where(classes == test_labels[i])[0][0]

    for i in validation_idxs:
        test_labels[i] = np.where(classes == test_labels[i])[0][0]
        
    train_images = torch.FloatTensor(train_images[train_idxs]).unsqueeze(1)
    train_labels = torch.LongTensor(train_labels[train_idxs])
    valid_images = torch.FloatTensor(test_images[validation_idxs]).unsqueeze(1)
    valid_labels = torch.LongTensor(test_labels[validation_idxs])
    test_images = torch.FloatTensor(test_images[test_idxs]).unsqueeze(1)
    test_labels = torch.LongTensor(test_labels[test_idxs])
    return train_images, train_labels, valid_images, valid_labels, test_images, \
                test_labels