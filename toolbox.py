"""
Coauthors: Michael Ainsworth
           Madi Kusmanov
           Yu-Chung Peng
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

    def __init__(self):
        super(SimpleCNN32Filter, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(144 * 32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144 * 32)
        x = self.fc1(x)
        return x


class SimpleCNN32Filter2Layers(nn.Module):
    """
    Define a simple CNN arhcitecture with 2 layers
    """

    def __init__(self):
        super(SimpleCNN32Filter2Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(12 * 12 * 32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN32Filter5Layers(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN32Filter5Layers, self).__init__()
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
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_rf_image(
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    fraction_of_train_samples,
    classes,
):
    """
    Peforms multiclass predictions for a random forest classifier
    """
    num_train_samples = int(np.sum(train_labels == 0) * fraction_of_train_samples)

    # Obtain only train images and labels for selected classes
    image_ls = []
    label_ls = []
    for cls in classes:
        cls_images = train_images[train_labels == cls]
        np.random.shuffle(cls_images)
        image_ls.append(cls_images[:num_train_samples])
        label_ls.append(np.repeat(cls, num_train_samples))

    train_images = np.concatenate(image_ls)
    train_labels = np.concatenate(label_ls)
    perm = np.random.permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

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


def run_dn_image(
    model,
    trainset,
    train_labels,
    testset,
    test_labels,
    fraction_of_train_samples,
    classes,
    epochs=5,
    lr=0.001,
    batch=16,
):
    """
    Peforms multiclass predictions for a deep network classifier
    """
    class_idxs = []
    for cls in classes:
        class_idx = np.argwhere(train_labels == cls).flatten()
        np.random.shuffle(class_idx)
        class_idx = class_idx[: int(len(class_idx) * fraction_of_train_samples)]
        class_idxs.append(class_idx)

    np.random.shuffle(class_idxs)

    train_idxs = np.concatenate(class_idxs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, num_workers=4, sampler=train_sampler
    )

    class_idxs = []
    for cls in classes:
        class_idx = np.argwhere(test_labels == cls).flatten()
        class_idxs.append(class_idx)

    test_idxs = np.concatenate(class_idxs)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=4, sampler=test_sampler
    )

    # define model
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # test the model
    correct = torch.tensor(0).to(dev)
    total = torch.tensor(0).to(dev)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = labels.clone().detach().to(dev)
            images = images.clone().detach().to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    accuracy = float(correct) / float(total)
    return accuracy
