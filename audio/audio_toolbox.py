"""
Coauthors: Yu-Chung Peng
           Haoyin Xu
           Madi Kusmanov
"""
import numpy as np
from sklearn.metrics import accuracy_score
import time
import torch
import os
import cv2
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as trans


def load_spoken_digit(path_recordings, feature_type="spectrogram"):
    file = os.listdir(path_recordings)

    audio_data = []  # audio data
    x_spec = []  # STFT spectrogram
    x_spec_mini = []  # resized image, 32*32
    y_number = []  # label of number
    y_speaker = []  # label of speaker
    if feature_type == "spectrogram":
        a = trans.Spectrogram(n_fft=128, normalized=True)
    elif feature_type == "melspectrogram":
        a = trans.MelSpectrogram(n_fft=128, normalized=True)
    elif feature_type == "mfcc":
        a = trans.MFCC(n_mfcc=128)
    for i in file:
        x, sr = librosa.load(path_recordings + i, sr=8000)
        x_stft_db = a(torch.tensor(x)).numpy()
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (32, 32))  # Resize into 32 by 32
        y_n = i[0]  # number
        y_s = i[2]  # first letter of speaker's name

        audio_data.append(x)
        x_spec.append(x_stft_db)
        x_spec_mini.append(x_stft_db_mini)
        y_number.append(y_n)
        y_speaker.append(y_s)

    x_spec_mini = np.array(x_spec_mini)
    y_number = np.array(y_number).astype(int)
    y_speaker = np.array(y_speaker)

    return x_spec_mini, y_number


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

    return accuracy_score(test_labels, test_preds), train_time, test_time


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
    start_time = time.perf_counter()
    for epoch in range(epochs):  # loop over the dataset multiple times

        for i in range(0, len(train_data), batch):
            # get the inputs
            inputs = train_data[i : i + batch].to(dev)
            labels = train_labels[i : i + batch].to(dev)
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
            for i in range(0, len(valid_data), batch):
                # get the inputs
                inputs = valid_data[i : i + batch].to(dev)
                labels = valid_labels[i : i + batch].to(dev)

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
    correct = torch.tensor(0).to(dev)
    total = torch.tensor(0).to(dev)
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(test_data), batch):
            inputs = test_data[i : i + batch].to(dev)
            labels = test_labels[i : i + batch].to(dev)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    end_time = time.perf_counter()
    test_time = end_time - start_time
    accuracy = float(correct) / float(total)
    return accuracy, train_time, test_time


def prepare_data(
    train_images, train_labels, test_images, test_labels, samples, classes
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
    return (
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        test_images,
        test_labels,
    )
