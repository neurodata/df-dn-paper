"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Jayanta Dey
"""
import time
import numpy as np
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleCNN32Filter(nn.Module):
    """
    Defines a simple CNN arhcitecture with 1 layer
    """

    def __init__(self, num_classes):
        super(SimpleCNN32Filter, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10, stride=2)
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
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


class SimpleCNN32Filter5Layers(nn.Module):
    """
    Define a simple CNN arhcitecture with 5 layers
    """

    def __init__(self, num_classes):
        super(SimpleCNN32Filter5Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8192, 200)
        self.fc2 = nn.Linear(200, num_classes)
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


def load_result(filename):
    """
    Loads results from specified file
    """
    inputs = open(filename, "r")
    lines = inputs.readlines()
    ls = []
    for line in lines:
        ls.append(float(line.strip()))
    return ls


def produce_mean(ls):
    """
    Produces means from list of 8 results
    """
    ls_space = []
    for i in range(int(len(ls) / 8)):
        l = ls[i * 8 : (i + 1) * 8]
        ls_space.append(l)

    return np.mean(ls_space, axis=0)


def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    """
    Return expected calibration error (ECE)
    """
    poba_hist = []
    accuracy_hist = []
    bin_size = 1 / num_bins
    total_sample = len(true_label)
    posteriors = predicted_posterior.max(axis=1)

    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (posteriors > bin * bin_size) & (posteriors <= (bin + 1) * bin_size)
        )[0]

        acc = (
            np.nan_to_num(np.mean(predicted_label[indx] == true_label[indx]))
            if indx.size != 0
            else 0
        )
        conf = np.nan_to_num(np.mean(posteriors[indx])) if indx.size != 0 else 0
        score += len(indx) * np.abs(acc - conf)

    score /= total_sample
    return score


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

    # Obtain only test images and labels for selected classes
    image_ls = []
    label_ls = []
    for cls in classes:
        image_ls.append(test_images[test_labels == cls])
        label_ls.append(np.repeat(cls, np.sum(test_labels == cls)))

    test_images = np.concatenate(image_ls)
    test_labels = np.concatenate(label_ls)

    # Train the model
    start_time = time.perf_counter()
    model.fit(train_images, train_labels)
    end_time = time.perf_counter()
    train_time = end_time - start_time

    # Test the model
    start_time = time.perf_counter()
    test_preds = model.predict(test_images)
    end_time = time.perf_counter()
    test_time = end_time - start_time

    test_probs = model.predict_proba(test_images)

    return (
        cohen_kappa_score(test_labels, test_preds),
        get_ece(test_probs, test_preds, test_labels),
        train_time,
        test_time,
    )


def run_dn_image_set(
    model,
    train_loader,
    test_loader,
    time_limit,
    ratio,
    lr=0.001,
    batch=64,
):
    """
    Peforms multiclass predictions for a deep network classifier
    """
    # define model
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    start_time = time.perf_counter()
    while True:  # loop over the dataset multiple times

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

        end_time = time.perf_counter()
        if (end_time - start_time) / ratio >= time_limit:
            train_time = end_time - start_time
            break

    # test the model
    model.eval()
    first = True
    prob_cal = nn.Softmax()
    start_time = time.perf_counter()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            test_labels = np.concatenate((test_labels, labels.tolist()))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if first:
                test_probs = test_prob.tolist()
                first = False
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    end_time = time.perf_counter()
    test_time = end_time - start_time
    return (
        cohen_kappa_score(test_preds, test_labels),
        get_ece(test_probs, test_preds, test_labels),
        train_time,
        test_time,
    )


def run_dn_image_es(
    model,
    train_loader,
    valid_loader,
    test_loader,
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

    start_time = time.perf_counter()
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
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

        # test generalization error for early stopping
        model.eval()
        cur_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.clone().detach().to(dev)
                labels = labels.clone().detach().to(dev)

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
    end_time = time.perf_counter()
    train_time = end_time - start_time

    # test the model
    model.eval()
    first = True
    prob_cal = nn.Softmax()
    start_time = time.perf_counter()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.clone().detach().to(dev)
            labels = labels.clone().detach().to(dev)
            test_labels = np.concatenate((test_labels, labels.tolist()))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if first:
                test_probs = test_prob.tolist()
                first = False
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    end_time = time.perf_counter()
    test_time = end_time - start_time
    return (
        cohen_kappa_score(test_preds, test_labels),
        get_ece(test_probs, test_preds, test_labels),
        train_time,
        test_time,
    )


def create_loaders_set(
    train_labels, test_labels, classes, trainset, testset, samples, batch=64
):
    """
    Creates training and testing loaders with fixed total samples
    """
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

    np.random.shuffle(class_idxs)

    train_idxs = np.concatenate(class_idxs)
    # change the labels to be from 0-len(classes)
    for i in train_idxs:
        trainset.targets[i] = np.where(classes == trainset.targets[i])[0][0]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, num_workers=4, sampler=train_sampler, drop_last=True
    )

    # get indicies of classes we want
    test_idxs = []
    for cls in classes:
        test_idx = np.argwhere(test_labels == cls).flatten()
        test_idxs.append(test_idx)

    test_idxs = np.concatenate(test_idxs)

    # change the labels to be from 0-len(classes)
    for i in test_idxs:
        testset.targets[i] = np.where(classes == testset.targets[i])[0][0]

    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        sampler=test_sampler,
        drop_last=True,
    )
    return train_loader, test_loader


def create_loaders_es(
    train_labels, test_labels, classes, trainset, testset, samples, batch=64
):

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

    np.random.shuffle(class_idxs)

    train_idxs = np.concatenate(class_idxs)
    # change the labels to be from 0-len(classes)
    for i in train_idxs:
        trainset.targets[i] = np.where(classes == trainset.targets[i])[0][0]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, num_workers=4, sampler=train_sampler, drop_last=True
    )

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
        testset.targets[i] = np.where(classes == testset.targets[i])[0][0]

    for i in validation_idxs:
        testset.targets[i] = np.where(classes == testset.targets[i])[0][0]

    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idxs)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        sampler=test_sampler,
        drop_last=True,
    )

    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_idxs)
    valid_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        sampler=valid_sampler,
        drop_last=True,
    )

    return train_loader, valid_loader, test_loader
