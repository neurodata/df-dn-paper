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
import torch.optim as optim


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
        image_ls.append(train_images[train_labels == cls][:num_train_samples])
        label_ls.append(np.repeat(cls, num_train_samples))

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
