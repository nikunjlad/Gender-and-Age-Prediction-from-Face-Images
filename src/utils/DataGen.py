from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, Dataset, sampler, SubsetRandomSampler, TensorDataset
import torch, h5py, os, cv2, random
from PIL import Image
import numpy as np


class Data(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(np.array(self.data[index], dtype=np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class DataGen:

    def __init__(self, config, logger):
        self.data = dict()
        self.data["age"] = dict()
        self.data["gender"] = dict()
        self.config = config
        self.logger = logger

    def load_data_from_h5(self, path):
        # loading data
        hf = h5py.File(path, 'r')
        # train, test data with labels being converted to numpy array from HDF5 format
        self.data["x_train"] = np.array(hf.get("x_train"), dtype=np.float32)
        self.data["x_test"] = np.array(hf.get("x_test"), dtype=np.float32)
        self.data["age"]["y_train"] = np.array(hf.get("y_train_age"), dtype=np.int64)
        self.data["age"]["y_test"] = np.array(hf.get("y_test_age"), dtype=np.int64)
        self.data["gender"]["y_train"] = np.array(hf.get("y_train_gender"), dtype=np.int64)
        self.data["gender"]["y_test"] = np.array(hf.get("y_test_gender"), dtype=np.int64)
        self.logger.debug("Training data: {}".format(str(self.data["x_train"].shape)))
        self.logger.debug("Testing data: {}".format(str(self.data["x_test"].shape)))
        self.logger.debug("Training labels: {}".format(str(self.data["age"]["y_train"].shape)))
        self.logger.debug("Testing labels: {}".format(str(self.data["age"]["y_test"].shape)))
        self.logger.debug("Training labels: {}".format(str(self.data["gender"]["y_train"].shape)))
        self.logger.debug("Testing labels: {}".format(str(self.data["gender"]["y_test"].shape)))
        self.logger.debug("Dataset read successfully!\n")

    def split_data(self):
        self.logger.debug("Splitting dataset...")
        valid_size = self.config["DATALOADER"]["VALIDATION_SPLIT"]  # % of data to be used for validation
        num_train = len(self.data["x_train"])  # get number of training samples
        indices = list(range(num_train))  # get indices of training data
        np.random.shuffle(indices)  # shuffle data randomly
        split = int(np.floor(valid_size * num_train))  # split threshold
        train_idx, valid_idx = indices[split:], indices[:split]  # split data
        x_train = self.data["x_train"][train_idx, :, :, :]
        x_valid = self.data["x_train"][valid_idx, :, :, :]
        y_train_age = self.data["age"]["y_train"][train_idx]
        y_valid_age = self.data["age"]["y_train"][valid_idx]
        y_train_gender = self.data["gender"]["y_train"][train_idx]
        y_valid_gender = self.data["gender"]["y_train"][valid_idx]

        # convert data to lists
        self.data["x_train"] = torch.tensor(x_train).permute(0, 3, 1, 2) # training data
        self.data["x_valid"] = torch.tensor(x_valid).permute(0, 3, 1, 2)  # validation data
        self.data["x_test"] = torch.tensor(self.data["x_test"]).permute(0, 3, 1, 2)  # test data
        self.data["age"]["y_train"] = torch.tensor(y_train_age, dtype=torch.int64)  # training age labels
        self.data["age"]["y_valid"] = torch.tensor(y_valid_age, dtype=torch.int64)  # validation age labels
        self.data["age"]["y_test"] = torch.tensor(self.data["age"]["y_test"], dtype=torch.int64)  # testing age labels
        self.data["gender"]["y_train"] = torch.tensor(y_train_gender, dtype=torch.int64)  # training gender labels
        self.data["gender"]["y_valid"] = torch.tensor(y_valid_gender, dtype=torch.int64)  # validation gender labels
        self.data["gender"]["y_test"] = torch.tensor(self.data["gender"]["y_test"], dtype=torch.int64)  # testing gender labels

    def configure_dataloaders(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Age Dataloaders
        self.data["age"]["train_dataset"] = Data(self.data["x_train"], self.data["age"]["y_train"],
                                                 transform=transform_train)
        self.data["age"]["valid_dataset"] = Data(self.data["x_valid"], self.data["age"]["y_valid"],
                                                 transform=transform_test)
        self.data["age"]["test_dataset"] = Data(self.data["x_test"], self.data["age"]["y_test"],
                                                transform=transform_test)
        self.data["age"]["train_dataloader"] = DataLoader(self.data["age"]["train_dataset"],
                                                          batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["age"]["valid_dataloader"] = DataLoader(self.data["age"]["valid_dataset"],
                                                          batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["age"]["test_dataloader"] = DataLoader(self.data["age"]["test_dataset"],
                                                         batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])

        # Gender Dataloaders
        self.data["gender"]["train_dataset"] = Data(self.data["x_train"], self.data["gender"]["y_train"],
                                                    transform=transform_train)
        self.data["gender"]["valid_dataset"] = Data(self.data["x_valid"], self.data["gender"]["y_valid"],
                                                    transform=transform_test)
        self.data["gender"]["test_dataset"] = Data(self.data["x_test"], self.data["gender"]["y_test"],
                                                   transform=transform_test)
        self.data["gender"]["train_dataloader"] = DataLoader(self.data["gender"]["train_dataset"],
                                                             batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["gender"]["valid_dataloader"] = DataLoader(self.data["gender"]["valid_dataset"],
                                                             batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["gender"]["test_dataloader"] = DataLoader(self.data["gender"]["test_dataset"],
                                                            batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
