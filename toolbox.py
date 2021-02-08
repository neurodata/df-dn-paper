"""
Coauthors: Michael Ainsworth
           Madi Kusmanov
           Yu-Chung Peng
           Haoyin Xu
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


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
