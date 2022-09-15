"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Adway Kanhere
"""
from toolbox import *

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms


logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_naive_rf():
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            cohen_kappa, ece, train_time, test_time = run_rf_image_set(
                RF,
                cifar_train_images,
                cifar_train_labels,
                cifar_test_images,
                cifar_test_labels,
                samples,
                classes,
            )
            naive_rf_kappa.append(cohen_kappa)
            naive_rf_ece.append(ece)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

    print("naive_rf finished")
    write_result(prefix + "naive_rf_kappa.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_tune_time = []
    cnn32_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32)
        for classes in classes_space:

            # Create loaders
            train_loader, valid_loader = create_loaders_ra(
                cifar_train_labels.copy(),
                cifar_valid_labels.copy(),
                classes,
                deepcopy(cifar_train_set),
                deepcopy(cifar_valid_set),
                samples,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32",
                train_loader,
                valid_loader,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            cnn32_final = SimpleCNN32Filter(len(classes))

            train_valid_loader, test_loader = create_loaders_ra(
                cifar_train_valid_labels.copy(),
                cifar_test_labels.copy(),
                classes,
                deepcopy(cifar_train_valid_set),
                deepcopy(cifar_test_set),
                samples,
            )

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_final,
                arm.parameters,
                train_valid_loader,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
            )

            cnn32_kappa.append(final_ck)
            cnn32_ece.append(final_ece)
            cnn32_tune_time.append(tune_time)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

            write_result(prefix + "cnn32_bestparams.txt", best_params)
            write_result(prefix + "cnn32_kappa.txt", cnn32_kappa)
            write_result(prefix + "cnn32_ece.txt", cnn32_ece)
            write_result(prefix + "cnn32_tune_time.txt", cnn32_tune_time)
            write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
            write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)
    print("cnn32 finished")


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_tune_time = []
    cnn32_2l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for classes in classes_space:

            # Create loaders
            train_loader, valid_loader = create_loaders_ra(
                cifar_train_labels.copy(),
                cifar_valid_labels.copy(),
                classes,
                deepcopy(cifar_train_set),
                deepcopy(cifar_valid_set),
                samples,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_2l",
                train_loader,
                valid_loader,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            cnn32_2l_final = SimpleCNN32Filter2Layers(len(classes))

            train_valid_loader, test_loader = create_loaders_ra(
                cifar_train_valid_labels.copy(),
                cifar_test_labels.copy(),
                classes,
                deepcopy(cifar_train_valid_set),
                deepcopy(cifar_test_set),
                samples,
            )

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_2l_final,
                arm.parameters,
                train_valid_loader,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
            )

            cnn32_2l_kappa.append(final_ck)
            cnn32_2l_ece.append(final_ece)
            cnn32_2l_tune_time.append(tune_time)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

            write_result(prefix + "cnn32_2l_bestparams.txt", best_params)
            write_result(prefix + "cnn32_2l_kappa.txt", cnn32_2l_kappa)
            write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
            write_result(prefix + "cnn32_2l_tune_time.txt", cnn32_2l_tune_time)
            write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
            write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)
    print("cnn32_2l finished")


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_tune_time = []
    cnn32_5l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for classes in classes_space:

            # Create loaders
            train_loader, valid_loader = create_loaders_ra(
                cifar_train_labels.copy(),
                cifar_valid_labels.copy(),
                classes,
                deepcopy(cifar_train_set),
                deepcopy(cifar_valid_set),
                samples,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_5l",
                train_loader,
                valid_loader,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            cnn32_5l_final = SimpleCNN32Filter5Layers(len(classes))

            train_valid_loader, test_loader = create_loaders_ra(
                cifar_train_valid_labels.copy(),
                cifar_test_labels.copy(),
                classes,
                deepcopy(cifar_train_valid_set),
                deepcopy(cifar_test_set),
                samples,
            )

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                cnn32_5l_final,
                arm.parameters,
                train_valid_loader,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
            )

            cnn32_5l_kappa.append(final_ck)
            cnn32_5l_ece.append(final_ece)
            cnn32_5l_tune_time.append(tune_time)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

            write_result(prefix + "cnn32_5l_bestparams.txt", best_params)
            write_result(prefix + "cnn32_5l_kappa.txt", cnn32_5l_kappa)
            write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
            write_result(prefix + "cnn32_5l_tune_time.txt", cnn32_5l_tune_time)
            write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
            write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)
    print("cnn32_5l finished")


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_tune_time = []
    resnet18_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (resnet18)
        for classes in classes_space:

            # Create loaders
            train_loader, valid_loader = create_loaders_ra(
                cifar_train_labels.copy(),
                cifar_valid_labels.copy(),
                classes,
                deepcopy(cifar_train_set),
                deepcopy(cifar_valid_set),
                samples,
            )

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "resnet18",
                train_loader,
                valid_loader,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            resnet_final = models.resnet18(pretrained=True)
            num_ftrs = resnet_final.fc.in_features
            resnet_final.fc = nn.Linear(num_ftrs, len(classes))

            train_valid_loader, test_loader = create_loaders_ra(
                cifar_train_valid_labels.copy(),
                cifar_test_labels.copy(),
                classes,
                deepcopy(cifar_train_valid_set),
                deepcopy(cifar_test_set),
                samples,
            )

            start_time = time.perf_counter()
            model_retrain_aftertune = train_net(
                resnet18_final,
                arm.parameters,
                train_valid_loader,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_loader, dev=device
            )

            resnet18_kappa.append(final_ck)
            resnet18_ece.append(final_ece)
            resnet18_tune_time.append(tune_time)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

            write_result(prefix + "resnet18_bestparams.txt", best_params)
            write_result(prefix + "resnet18_kappa.txt", resnet18_kappa)
            write_result(prefix + "resnet18_ece.txt", resnet18_ece)
            write_result(prefix + "resnet18_tune_time.txt", resnet18_tune_time)
            write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
            write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)
    print("resnet18 finished")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_10.py -m 3
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(10, 10000, num=8, dtype=int)

    nums = list(range(10))
    np.random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    # scale = np.mean(np.arange(0, 256))
    # normalize = lambda x: (x - scale) / scale

    # # train data
    # cifar_train_set = datasets.CIFAR10(
    #     root="./", train=True, download=True, transform=None
    # )
    # cifar_train_images = normalize(cifar_train_set.data)
    # cifar_train_labels = np.array(cifar_train_set.targets)

    # # test data
    # cifar_test_set = datasets.CIFAR10(
    #     root="./", train=False, download=True, transform=None
    # )
    # cifar_test_images = normalize(cifar_test_set.data)
    # cifar_test_labels = np.array(cifar_test_set.targets)

    # cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
    # cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)

    # run_naive_rf()

    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # train data
    cifar_train_set = datasets.CIFAR10(
        root="./", train=True, download=True, transform=data_transforms
    )

    # test data
    cifar_test_set = datasets.CIFAR10(
        root="./", train=False, download=True, transform=data_transforms
    )

    # Combine all data into whole set
    cifar_whole_images = np.concatenate((cifar_train_set.data, cifar_test_set.data))
    cifar_whole_labels = np.concatenate(
        (np.array(cifar_train_set.targets), np.array(cifar_test_set.targets))
    )

    # Separate whole set into training set & valid set & test set with 2:1:1 ratio
    (
        cifar_train_valid_images,
        cifar_test_images,
        cifar_train_valid_labels,
        cifar_test_labels,
    ) = train_test_split(
        cifar_whole_images,
        cifar_whole_labels,
        test_size=0.25,
        stratify=cifar_whole_labels,
    )
    (
        cifar_valid_images,
        cifar_train_images,
        cifar_valid_labels,
        cifar_train_labels,
    ) = train_test_split(
        cifar_train_valid_images,
        cifar_train_valid_labels,
        test_size=0.67,
        stratify=cifar_train_valid_labels,
    )

    # Create new datasets
    cifar_train_valid_set = deepcopy(cifar_train_set)
    cifar_train_valid_set.data = cifar_train_valid_images
    cifar_train_valid_set.targets = cifar_train_valid_labels

    cifar_valid_set = deepcopy(cifar_train_set)
    cifar_valid_set.data = cifar_valid_images
    cifar_valid_set.targets = cifar_valid_labels

    cifar_test_set = deepcopy(cifar_train_set)
    cifar_test_set.data = cifar_test_images
    cifar_test_set.targets = cifar_test_labels

    cifar_train_set.data = cifar_train_images
    cifar_train_set.targets = cifar_train_labels

    run_cnn32()
    run_cnn32_2l()
    run_cnn32_5l()

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # train data
    cifar_train_set = datasets.CIFAR10(
        root="./", train=True, download=True, transform=data_transforms
    )

    # test data
    cifar_test_set = datasets.CIFAR10(
        root="./", train=False, download=True, transform=data_transforms
    )

    # Combine all data into whole set
    cifar_whole_images = np.concatenate((cifar_train_set.data, cifar_test_set.data))
    cifar_whole_labels = np.concatenate(
        (np.array(cifar_train_set.targets), np.array(cifar_test_set.targets))
    )

    # Separate whole set into training set & valid set & test set with 2:1:1 ratio
    (
        cifar_train_valid_images,
        cifar_test_images,
        cifar_train_valid_labels,
        cifar_test_labels,
    ) = train_test_split(
        cifar_whole_images,
        cifar_whole_labels,
        test_size=0.25,
        stratify=cifar_whole_labels,
    )
    (
        cifar_valid_images,
        cifar_train_images,
        cifar_valid_labels,
        cifar_train_labels,
    ) = train_test_split(
        cifar_train_valid_images,
        cifar_train_valid_labels,
        test_size=0.67,
        stratify=cifar_train_valid_labels,
    )

    # Create new datasets
    cifar_train_valid_set = deepcopy(cifar_train_set)
    cifar_train_valid_set.data = cifar_train_valid_images
    cifar_train_valid_set.targets = cifar_train_valid_labels

    cifar_valid_set = deepcopy(cifar_train_set)
    cifar_valid_set.data = cifar_valid_images
    cifar_valid_set.targets = cifar_valid_labels

    cifar_test_set = deepcopy(cifar_train_set)
    cifar_test_set.data = cifar_test_images
    cifar_test_set.targets = cifar_test_labels

    cifar_train_set.data = cifar_train_images
    cifar_train_set.targets = cifar_train_labels

    run_resnet18()
