import torch, warnings, torchvision, os, json, time, yaml, datetime, logging, argparse, sys
from utils.DataGen import DataGen
from model import AgeNet, GenderNet
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True


class Main(DataGen):

    def __init__(self, args):
        # loading the YAML configuration file
        self.args = args  # user configurable parameters
        with open("../configs/config.yaml", 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")  # get current datetime
        if not os.path.exists(self.config["DATA"]["OUTPUT_DIR"]):
            os.mkdir(self.config["DATA"]["OUTPUT_DIR"])
        if not os.path.exists("logs"):
            os.mkdir("logs")  # make log directory if does not exist
        os.chdir("logs")  # change to logs directory
        # getting the custom logger
        self.logger_name = "face_" + self.current_time + "_.log"
        self.logger = self.get_loggers(self.logger_name)
        self.logger.info("Age and Gender Inference!")
        self.logger.info("Current time: " + str(self.current_time))
        self.train_on_gpu = self.config["GPU"]["STATUS"]
        DataGen.__init__(self, self.config, self.logger)
        os.chdir("..")  # change directory to base path

    @staticmethod
    def get_loggers(name):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        f_hand = logging.FileHandler(name)  # file where the custom logs needs to be handled
        f_hand.setLevel(logging.DEBUG)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(f_format)
        logger.addHandler(ch)

        return logger

    # checking if cuda is available
    def configure_cuda(self, device_id):
        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu and self.config["GPU"]["STATUS"]:
            self.logger.info('Training on CPU ...')
        else:
            torch.cuda.set_device(device_id)
            self.logger.info('CUDA is available! Training on {} NVidia {} GPUs'.format(
                str(len(self.config["GPU"]["DEVICES"])), str(torch.cuda.get_device_name(0))))

    def train(self, net, epochs, optimizer, criterion, scheduler, stats, args, output_path):

        history = list()
        train_start = time.time()  # start_time of training
        best_val_loss = float('inf')  # initially loss in infinite
        model_name = self.args.age_gender + "_model_" + str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"]) + "_" + \
                     str(len(self.config["GPU"]["DEVICES"])) + ".pt"

        # for all the epochs
        for epoch in range(epochs):
            epoch_start = time.time()  # start time for the epoch
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            # Set to training mode
            net.train()

            # Loss and Accuracy within the epoch, initial values are set to be 0
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(tqdm(self.data[args.age_gender]["train_dataloader"])):

                scheduler.step()  # stepping through the learning rate for optimal convergence

                # if GPU mentioned.
                if self.train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()  # Clean existing gradients
                outputs = net(inputs)  # Forward pass - compute outputs on input data using the model
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagate the gradients
                optimizer.step()  # Update the parameters

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)  # Compute the accuracy
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                # print("Batch: {:03d}/{:03d}, Training Loss: {:.4f}, "
                #       "Training Acc: {:.4f}".format(i, stats["data"]["training"]["num_batches"], loss.item(),
                #                                     acc.item() * 100))

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                net.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(tqdm(self.data[args.age_gender]["valid_dataloader"])):

                    if self.train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    outputs = net(inputs)  # Forward pass - compute outputs on input data using the model
                    loss = criterion(outputs, labels)  # Compute loss

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)
                    ret, predictions = torch.max(outputs.data, 1)  # Calculate validation accuracy
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    # print("Validation Batch number: {:03d}/{:03d}, Validation Loss: {:.4f}, "
                    #       "Validation Acc: {:.4f}".format(j, stats["data"]["validation"]["num_batches"], loss.item(),
                    #                                       acc.item() * 100))

            # resetting scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, stats["data"]["training"]["num_batches"],
                                                             eta_min=0)

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / stats["data"]["training"]["num_samples"]
            avg_train_acc = train_acc / float(stats["data"]["training"]["num_samples"])

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / stats["data"]["validation"]["num_samples"]
            avg_valid_acc = valid_acc / float(stats["data"]["validation"]["num_samples"])

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()
            print("-" * 89)
            print("Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}%, Valid Loss : {:.4f}, "
                  "Valid Acc: {:.4f}%, Time: {:.4f}s".format(epoch + 1, avg_train_loss, avg_train_acc * 100,
                                                             avg_valid_loss, avg_valid_acc * 100,
                                                             epoch_end - epoch_start))
            print("-" * 89)

            if avg_valid_loss < best_val_loss:
                print("\nPrevious Best loss: {:.4f} | New Best Loss: {:.4f} | "
                      "Saving Best model...\n".format(best_val_loss, avg_valid_loss))
                torch.save(net.state_dict(), output_path + "/" + model_name)
                best_val_loss = avg_valid_loss  # new best loss is the recently found validation loss

        exec_time = time.time() - train_start
        self.logger.info("Time taken for training: {}".format(str(exec_time)))

        return net, history, exec_time, model_name

    def test(self, net, criterion, output_path, model_name, stats, args):

        # load model after training for testing
        net.load_state_dict(torch.load(output_path + "/" + model_name))

        test_loss = 0
        test_acc = 0
        test_hist = list()

        # Validation - No gradient tracking needed
        with torch.no_grad():
            net.eval()  # Set to evaluation mode

            # Validation loop
            for j, (inputs, labels) in enumerate(tqdm(self.data[args.age_gender]["test_dataloader"])):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = net(inputs)  # Forward pass - compute outputs on input data using the model
                loss = criterion(outputs, labels)  # Compute loss

                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                print(predictions.cpu().numpy()[0])
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                test_acc += acc.item() * inputs.size(0)

                # print("Test Batch number: {:03d}/{:03d}, Test Loss: {:.4f}, "
                #       "Test Accuracy: {:.4f}".format(j, stats["data"]["testing"]["num_batches"], loss.item(),
                #                                      acc.item() * 100))

            avg_test_loss = test_loss / stats["data"]["testing"]["num_samples"]
            avg_test_acc = test_acc / float(stats["data"]["testing"]["num_samples"])

            test_hist.append([avg_test_loss, avg_test_acc])
            print("Test: Loss : {:.4f}, Accuracy: {:.4f}%".format(avg_test_loss, avg_test_acc * 100))

        return test_hist

    @staticmethod
    def plot_graphs(hist, epochs, x_label, y_label, plt_title, legend, save_name):

        plt.figure(figsize=(7, 6))
        x = np.array([i for i in range(0, epochs)])
        if y_label == "Accuracy":
            plt.plot(x, hist[:, 2])
            plt.plot(x, hist[:, 3])
        else:
            plt.plot(x, hist[:, 0])
            plt.plot(x, hist[:, 1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plt_title)
        plt.legend(legend, loc='upper right')
        plt.savefig(save_name)

    def export_stats(self, stats, history, test_hist, exec_time, output_path):

        stats["hyperparameters"] = dict()
        stats["device"] = dict()
        stats["device"]["type"] = ["gpu" if self.config["GPU"]["STATUS"] else "cpu"][0]
        stats["device"]["parallel"] = self.config["GPU"]["PARALLEL"]
        stats["device"]["devices"] = list()
        if stats["device"]["type"] == "gpu":
            if isinstance(self.config["GPU"]["DEVICES"], list):
                for dev in self.config["GPU"]["DEVICES"]:
                    info = dict()
                    info["id"] = dev
                    info["device_name"] = torch.cuda.get_device_properties(dev).name
                    info["total_memory (MB)"] = torch.cuda.get_device_properties(dev).total_memory * (2 ** -20)
                    stats["device"]["devices"].append(info)
            else:
                info = dict()
                info["id"] = self.config["GPU"]["DEVICES"][0]
                info["device_name"] = torch.cuda.get_device_properties(info["id"]).name
                info["total_memory (MB)"] = torch.cuda.get_device_properties(info["id"]).total_memory * (2 ** -20)
                stats["device"]["devices"].append(info)
        stats["hyperparameters"]["epochs"] = self.config["HYPERPARAMETERS"]["EPOCHS"]
        stats["hyperparameters"]["learning_rate"] = self.config["HYPERPARAMETERS"]["OPTIMIZER"]["LR"]
        stats["hyperparameters"]["batch_size"] = self.config["HYPERPARAMETERS"]["BATCH_SIZE"]
        stats["hyperparameters"]["optimizer"] = self.config["HYPERPARAMETERS"]["OPTIMIZER"]["NAME"]
        stats["metrics"] = dict()
        stats["metrics"]["training_loss"] = history[-1][0]
        stats["metrics"]["training_accuracy"] = history[-1][2]
        stats["metrics"]["validation_loss"] = history[-1][1]
        stats["metrics"]["validation_accuracy"] = history[-1][3]
        stats["metrics"]["test_loss"] = test_hist[-1][0]
        stats["metrics"]["test_accuracy"] = test_hist[-1][1]
        stats["metrics"]["runtime (secs)"] = exec_time
        stats["training_history"] = history
        stats["test_history"] = test_hist

        stats_name = self.args.age_gender + "_stats_" + str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"]) + "_" + \
                     str(len(self.config["GPU"]["DEVICES"])) + ".json"
        with open(output_path + "/" + stats_name, 'w') as outfile:
            json.dump(stats, outfile, indent=2)

    def main(self):
        """
        Main function for program execution
        :return:
        """
        stats = dict()  # to capture running statistics

        # 1. configuring GPU
        # configure GPU if available
        if self.config["GPU"]["STATUS"]:
            if self.config["GPU"]["DEVICES"] is not None:
                self.configure_cuda(self.config["GPU"]["DEVICES"][0])

        # 2. configuring paths
        # configure data path
        if os.getenv("HOME") != self.config["DATA"]["DATA_DIR"]:
            self.config["DATA"]["DATA_DIR"] = os.getenv("HOME")
        dir_name = str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"]) + "_" + self.config["DATA"]["OUTPUT_DIR"] + "_" + \
                   str(len(self.config["GPU"]["DEVICES"]))
        output_path = os.path.join(self.config["DATA"]["OUTPUT_DIR"], dir_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # 3. configuring target labels, in our case we have 2 classification tasks, gender and age classification
        ages = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(21, 24)", "(25, 32)",
                "(33, 37)", "(38, 43)", "(44, 47)", "(48, 53)", "(54, 59)", "(60, 100)"]
        genders = ["Male", "Female"]

        # 4. loading data
        data_path = os.path.join(self.config["DATA"]["DATA_DIR"], "data", self.config["DATALOADER"]["DATASET_NAME"],
                                 "adience.h5")
        self.load_data_from_h5(data_path)
        self.split_data()
        self.configure_dataloaders()

        # 5. getting dataloader information, batch size and sample counts
        # since age and gender both are split using same validation ratio, their sizes will be same.
        train_data_size = len(self.data["age"]["train_dataset"])
        valid_data_size = len(self.data["age"]["valid_dataset"])
        test_data_size = len(self.data["age"]["test_dataset"])
        num_train_data_batches = len(self.data["age"]["train_dataloader"])
        num_valid_data_batches = len(self.data["age"]["valid_dataloader"])
        num_test_data_batches = len(self.data["age"]["test_dataloader"])

        # 6. update stats of data information
        stats["data"] = dict()
        train_dict = dict()
        train_dict["num_samples"] = train_data_size
        train_dict["num_batches"] = num_train_data_batches
        stats["data"]["training"] = train_dict
        valid_dict = dict()
        valid_dict["num_samples"] = valid_data_size
        valid_dict["num_batches"] = num_valid_data_batches
        stats["data"]["validation"] = valid_dict
        test_dict = dict()
        test_dict["num_samples"] = test_data_size
        test_dict["num_batches"] = num_test_data_batches
        stats["data"]["testing"] = test_dict

        # 7. display batch information
        self.logger.info("Number of training samples: {}".format(str(train_data_size)))
        self.logger.info("{} batches each having {} samples".format(str(num_train_data_batches),
                                                                    str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"])))
        self.logger.info("Number of validation samples: {}".format(str(valid_data_size)))
        self.logger.info("{} batches each having {} samples".format(str(num_valid_data_batches),
                                                                    str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"])))
        self.logger.info("Number of testing samples: {}".format(str(test_data_size)))
        self.logger.info("{} batches each having {} samples".format(str(num_test_data_batches),
                                                                    str(self.config["HYPERPARAMETERS"]["BATCH_SIZE"])))

        # 8. export a grid of images or exploring our data visually
        batch = next(iter(self.data[args.age_gender]["train_dataloader"]))
        images, labels = batch

        if self.config["HYPERPARAMETERS"]["PLOT_IMG"]:
            grid = torchvision.utils.make_grid(images[:64], nrow=8)
            plt.figure(figsize=(10, 10))
            np.transpose(grid, (1, 2, 0))
            save_image(grid, 'grid.png')
            for data, target in self.data[args.age_gender]["train_dataloader"]:
                self.logger.debug("Batch image tensor dimensions: {}".format(str(data.shape)))
                self.logger.debug("Batch label tensor dimensions: {}".format(str(target.shape)))
                break

        # 9. loading Network based on the task at hand
        task = lambda x: AgeNet() if x == "age" else GenderNet()
        net = task(self.args.age_gender)

        # 10. use GPU if available else cpu by default
        if self.train_on_gpu:
            net = net.cuda()
        self.logger.debug(str(net))

        # 11. data parallel mode if enabled by user for parallel training
        if self.config["GPU"]["PARALLEL"]:
            net = torch.nn.DataParallel(net, device_ids=self.config["GPU"]["DEVICES"])
            cudnn.benchmark = True

        # 12. optimizers and loss functions and scheduler for auto adjusting learning rate based on performance
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.config["HYPERPARAMETERS"]["OPTIMIZER"]["LR"],
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_data_batches, eta_min=0)

        # 13. training our model
        epochs = self.config["HYPERPARAMETERS"]["EPOCHS"]
        net, history, exec_time, model_name = self.train(net, self.config["HYPERPARAMETERS"]["EPOCHS"], optimizer,
                                                         criterion, scheduler, stats, args, output_path)

        hist = np.array(history)  # convert history from list to numpy array

        # 14. plot training-validation accuracy and loss curves
        loss_name = output_path + "/" + self.args.age_gender + "_train_valid_loss_" + str(len(self.config["GPU"]["DEVICES"])) + ".png"
        acc_name = output_path + "/" + self.args.age_gender + "_train_valid_accuracy_" + str(len(self.config["GPU"]["DEVICES"])) + ".png"

        self.plot_graphs(hist, epochs, x_label="Epochs", y_label="Cross-Entropy Loss", plt_title="Loss Curves",
                         legend=['train_loss', 'valid_loss'], save_name=loss_name)
        self.plot_graphs(hist, epochs, x_label="Epochs", y_label="Accuracy", plt_title="Accuracy Curves",
                         legend=['train_acc', 'valid_acc'], save_name=acc_name)

        # 15. testing
        test_hist = self.test(net, criterion, output_path, model_name, stats, args)

        # 16. exporting statistics
        self.export_stats(stats, history, test_hist, exec_time, output_path)

        # terminating program
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Age and Gender Inference')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to input image or video file. Skip this argument to capture frames from a '
                             'camera.')
    parser.add_argument('-ag', "--age-gender", type=str, required=True,
                        default="age", help="mention classification needs to be performed - age or gender")
    args = parser.parse_args()
    m = Main(args)
    m.main()
