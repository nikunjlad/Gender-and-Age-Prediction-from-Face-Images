from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, Dataset, sampler, SubsetRandomSampler, TensorDataset
import torch, h5py, os, cv2, random
from PIL import Image
import numpy as np


class Data(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class DataGen:

    def __init__(self, config, logger):
        self.data = dict()
        self.config = config
        self.logger = logger

    def load_data_from_h5(self, path):
        # loading data
        hf = h5py.File(path, 'r')
        # train, test data with labels being converted to numpy array from HDF5 format
        self.data["x_train"] = np.array(hf.get("X_train"), dtype=np.float32)
        self.data["x_test"] = np.array(hf.get("X_test"), dtype=np.float32)
        self.data["y_train_age"] = np.array(hf.get("y_train_age"), dtype=np.int64)
        self.data["y_test_age"] = np.array(hf.get("y_test_age"), dtype=np.int64)
        self.data["y_train_gender"] = np.array(hf.get("y_train_gender"), dtype=np.int64)
        self.data["y_test_gender"] = np.array(hf.get("y_test_gender"), dtype=np.int64)
        self.logger.debug("Training data: {}".format(str(self.data["x_train"].shape)))
        self.logger.debug("Testing data: {}".format(str(self.data["x_test"].shape)))
        self.logger.debug("Training labels: {}".format(str(self.data["y_train_age"].shape)))
        self.logger.debug("Testing labels: {}".format(str(self.data["y_test_age"].shape)))
        self.logger.debug("Training labels: {}".format(str(self.data["y_train_gender"].shape)))
        self.logger.debug("Testing labels: {}".format(str(self.data["y_test_gender"].shape)))

    def load_data_from_dirs(self, data_dir, categories):

        """
            Citations: Python documentation was referred for understanding directory traversing using os
            :param filename: the current filename - in this case it point to this file -> signs1.py
            :return: current direcotory, train data and test data directory paths
            """
        data = list()  # a list of lists. Each tuple is a [image, label] format

        # loop over all the categories
        for index, category in enumerate(categories):
            path = os.path.join(data_dir, category)  # path to every alphabet
            # path name leading to the alphabet directory
            self.logger.info("Opening directory {} from path: {}".format(str(category), str(path)))

            # loop over all the images in the directory
            for img in os.listdir(path):
                try:
                    # reading the image in original format
                    image = cv2.imread(os.path.join(path, img))
                    data.append([image, index])  # append the [image, label] list in the data list
                except Exception as e:
                    print(e)

        random.shuffle(data)  # randomly shuffling data

        return data

    def split_data(self):
        valid_size = self.config["DATALOADER"]["VALIDATION_SPLIT"]  # % of data to be used for validation
        num_train = len(self.data["x_train"])  # get number of training samples
        indices = list(range(num_train))  # get indices of training data
        np.random.shuffle(indices)  # shuffle data randomly
        split = int(np.floor(valid_size * num_train))  # split threshold
        train_idx, valid_idx = indices[split:], indices[:split]  # split data
        X_train = self.data["x_train"][train_idx, :, :, :]
        X_valid = self.data["x_train"][valid_idx, :, :, :]
        y_train_age = self.data["y_train_age"][train_idx]
        y_valid_age = self.data["y_train_age"][valid_idx]
        y_train_gender = self.data["y_train_gender"][train_idx]
        y_valid_gender = self.data["y_train_gender"][valid_idx]

        # convert data to lists
        self.data["x_train"] = list(X_train.transpose(0, 3, 1, 2))  # training data
        self.data["x_valid"] = list(X_valid.transpose(0, 3, 1, 2))  # validation data
        self.data["x_test"] = list(self.data["x_test"].transpose(0, 3, 1, 2))   # test data
        self.data["y_train_age"] = list(y_train_age)  # training age labels
        self.data["y_valid_age"] = list(y_valid_age)  # validation age labels
        self.data["y_train_gender"] = list(y_train_gender)  # training gender labels
        self.data["y_valid_gender"] = list(y_valid_gender)  # validation gender labels
        self.data["y_test_age"] = list(self.data["y_test_age"])  # testing age labels
        self.data["y_test_gender"] = list(self.data["y_test_gender"])  # testing gender labels

    def configure_dataloaders(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.data["train_dataset"] = Data(self.data["x_train"], self.data["y_train"], transform=transform_train)
        self.data["valid_dataset"] = Data(self.data["x_valid"], self.data["y_valid"], transform=transform_test)
        self.data["test_dataset"] = Data(self.data["x_test"], self.data["y_test"], transform=transform_test)
        self.data["train_dataloader"] = DataLoader(self.data["train_dataset"],
                                                   batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["valid_dataloader"] = DataLoader(self.data["valid_dataset"],
                                                   batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["test_dataloader"] = DataLoader(self.data["test_dataset"],
                                                  batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
