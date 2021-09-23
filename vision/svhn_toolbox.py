"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score

import torch

from toolbox import (
    SimpleCNN32Filter,
    SimpleCNN32Filter2Layers,
    SimpleCNN32Filter5Layers,
    write_result,
    combinations_45,
    load_result,
    produce_mean,
    run_rf_image_set,
    run_dn_image_es,
)


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
        trainset.labels[i] = np.where(classes == trainset.labels[i])[0][0]

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
        testset.labels[i] = np.where(classes == testset.labels[i])[0][0]

    for i in validation_idxs:
        testset.labels[i] = np.where(classes == testset.labels[i])[0][0]

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
