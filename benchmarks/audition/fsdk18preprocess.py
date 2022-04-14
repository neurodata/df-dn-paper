import pandas as pd
import os


def preprocessdataset(data_folder, labels_file):
    """
    This function performs pre-preocessing of the FSDKaggle18 dataset such that:
    - Of all the labels, select only those which contain 300 samples per label
    - Extract the corresponding audio files for the selected labels
    --------------------------------------------
    Input parameters:
        data_folder: str
          Path to the folder containing the raw audio files
        labels_file: str
          Path to the csv file containing the labels
    Returns:
        path_recordings: list
          list containing the full path for each selected raw audio file
        labels_chosen: pandas DataFrame
          DataFrame of ground truth labels and metadata of all the selected raw audio files
        get_labels: array
          array of encoded unique label numbers
    """

    train_folder = data_folder
    train_label = pd.read_csv(labels_file)

    # select subset of data that only contains 300 samples per class
    labels_chosen = train_label[
        train_label["label"].map(train_label["label"].value_counts() == 300)
    ]

    training_files = []
    for file in os.listdir(train_folder):
        for x in labels_chosen.fname.to_list():
            if file.endswith(x):
                training_files.append(file)

    path_recordings = []
    for audiofile in training_files:
        path_recordings.append(os.path.join(train_folder, audiofile))

    # convert selected label names to integers
    labels_to_index = {
        "Acoustic_guitar": 0,
        "Applause": 1,
        "Bass_drum": 2,
        "Trumpet": 3,
        "Clarinet": 4,
        "Double_bass": 5,
        "Laughter": 6,
        "Shatter": 7,
        "Snare_drum": 8,
        "Saxophone": 9,
        "Tearing": 10,
        "Flute": 11,
        "Hi-hat": 12,
        "Violin_or_fiddle": 13,
        "Squeak": 14,
        "Fart": 15,
        "Fireworks": 16,
        "Cello": 17,
    }

    # encode labels to integers
    get_labels = labels_chosen["label"].replace(labels_to_index).to_list()
    labels_chosen = labels_chosen.reset_index()

    return path_recordings, labels_chosen, get_labels
