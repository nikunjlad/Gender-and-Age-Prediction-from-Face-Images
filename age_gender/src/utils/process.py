import h5py, cv2, os, random, argparse
from tqdm import tqdm
import numpy as np


class Process:

    def __init__(self, data_path=None, filename=None):
        self.filename = filename
        self.data_path = data_path
        self.prefix = "landmark_aligned_face."  # every image name is prefixed with this string

        # 5 folders to loop over, each folder text file contains information of other folders
        self.folder_files = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt',
                             'fold_4_data.txt']

        # age category classes, there are 12 age groups
        self.ages = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(21, 24)", "(25, 32)",
                     "(33, 37)", "(38, 43)", "(44, 47)", "(48, 53)", "(54, 59)", "(60, 100)"]

        # there are only 2 gender categories
        self.genders = ['m', 'f']

        # Since there are labels that do not match the classes stated, need to fix them
        self.ages_to_fix = {'35': self.ages[6], '3': self.ages[0], '55': self.ages[10], '58': self.ages[10],
                            '22': self.ages[4], '13': self.ages[2], '45': self.ages[8], '36': self.ages[6],
                            '23': self.ages[4], '57': self.ages[10], '56': self.ages[10], '2': self.ages[0],
                            '29': self.ages[5], '34': self.ages[6], '42': self.ages[7], '46': self.ages[8],
                            '32': self.ages[5], '(38, 48)': self.ages[7], '(38, 42)': self.ages[7],
                            '(8, 23)': self.ages[2], '(27, 32)': self.ages[5]}

        self.none_count = 0
        self.no_age = 0

    def get_image_paths(self, folder_file):

        # one big folder list
        folder = list()
        folder_path = os.path.join(self.data_path, folder_file)

        # start processing each folder text file
        with open(folder_path) as text:
            lines = text.readlines()
            print("Total lines to be parsed from this document: ", len(lines))

            # loop over all the lines ignoring the first line which contains metadata of the file contents
            for line in lines[1:]:
                line = line.strip().split("\t")  # strip tab character from each line

                # line[0] contains folder name, line[2] gives information of image id, line[1] gives exact image name
                # construct image path with above information
                img_path = line[0] + "/" + self.prefix + line[2] + "." + line[1]  # real image path

                # if the age group is not provided, and it is None, then increment None counter and continue to next
                # image. Likewise, check if the gender is provided or not, if not then just continue
                if line[3] == "None":
                    self.none_count += 1
                    continue

                if line[4] == "u" or line[4] == "":
                    self.no_age += 1
                    continue

                # We store useful metadata infos. for every right image, append the image along with
                folder.append([img_path] + line[3:5])
                if folder[-1][1] in self.ages_to_fix:
                    folder[-1][1] = self.ages_to_fix[folder[-1][1]]

        random.shuffle(folder)

        return folder

    def imread(self, path, width, height):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return img

    def aggregate_data(self, all_folders):

        width, height = 227, 227

        # loop for reading imgs from five folders
        all_data = []
        all_ages = []
        all_genders = []
        print("Start reading images data...")
        for ind, folder in enumerate(all_folders):
            data = []
            ages = []
            genders = []
            for i in tqdm(range(len(folder))):  # here using tqdm to monitor progress
                img_path = self.data_path + os.path.join("/aligned", folder[i][0])
                img = self.imread(img_path, width, height)
                data.append(img)
                ages.append(self.ages.index(folder[i][1]))
                genders.append(self.genders.index(folder[i][2]))
            all_data.append(data)
            all_ages.append(ages)
            all_genders.append(genders)
            print("Finished processing folder {}".format(str(ind)))

        print("All done!")
        all_data = np.concatenate(all_data)
        all_ages = np.concatenate(all_ages)
        all_genders = np.concatenate(all_genders)
        return all_data, all_ages, all_genders

    def split_data_from_dirs(self, data, ages, genders, split):
        """
        this function takes in data, labels and % of training data to be used. since % of data for training varies based on
        applications we keep that parameter user configurable.
        :param data: 4D numpy array of images in (num samples, width, height, channels) format
        :param labels: 1D numpy array storing labels for corresponding images
        :param split: percentage of data to be used for training
        :return:  return the splits of training and testing along with labels
        """
        print("Number of images in the training data: {}".format(str(data.shape[0])))
        print("Ages/Genders: {}".format(str(ages.shape)))

        # multiply split percentage with total images length and floor the result. Also cast into int, for slicing array
        split_factor = int(np.floor(split * data.shape[0]))  # number of images to be kept in training data
        print("Using {} images for training and {} images for testing!".format(str(split_factor),
                                                                               str(data.shape[0] - split_factor)))
        x_train = data[:split_factor, :, :, :].astype("float")
        x_test = data[split_factor:, :, :, :].astype("float")
        y_train_age = ages[:split_factor]
        y_test_age = ages[split_factor:]
        y_train_gender = genders[:split_factor]
        y_test_gender = genders[split_factor:]

        print("Training data shape: {}".format(str(x_train.shape)))
        print("Testing data shape: {}".format(str(x_test.shape)))
        print("Training Age labels shape: {}".format(str(y_train_age.shape)))
        print("Testing Age labels shape: {}".format(str(y_test_age.shape)))
        print("Training Gender labels shape: {}".format(str(y_train_gender.shape)))
        print("Testing Gender labels shape: {}".format(str(y_test_gender.shape)))

        return x_train, x_test, y_train_age, y_test_age, y_train_gender, y_test_gender

    def generate_h5(self, Xtr, Xtst, ytr_age, ytst_age, ytr_gen, ytst_gen):
        print("Generating H5 file...")
        hf = h5py.File(self.filename, 'w')
        hf.create_dataset('x_train', data=Xtr, compression="gzip")
        hf.create_dataset('x_test', data=Xtst, compression="gzip")
        hf.create_dataset('y_train_age', data=ytr_age, compression="gzip")
        hf.create_dataset('y_test_age', data=ytst_age, compression="gzip")
        hf.create_dataset('y_train_gender', data=ytr_gen, compression="gzip")
        hf.create_dataset('y_test_gender', data=ytst_gen, compression="gzip")
        hf.close()
        print("H5 file generated successfully")

    def helper(self):

        # looping over all the folder text files to aggregate the image paths
        all_folders = []
        for folder_file in self.folder_files:
            folder = self.get_image_paths(folder_file)
            all_folders.append(folder)
        # print("A sample:", all_folders[0][0])
        print("No. of Pics without Age Group Label:", self.none_count)

        # total data received after aggregating
        data, ages, genders = self.aggregate_data(all_folders)
        print("Aggregated data shape: {}".format(str(data.shape)))
        print("Aggregated age shape: {}".format(str(ages.shape)))
        print("Aggregated genders shape: {}".format(str(genders.shape)))

        # splitting data into training and testing based on percentage. split is amount of training data to be used
        split = 0.95
        x_train, x_test, y_train_age, y_test_age, y_train_gender, y_test_gender = self.split_data_from_dirs(data, ages,
                                                                                                            genders,
                                                                                                            split)

        # encapsulating data into h5 files
        self.generate_h5(x_train, x_test, y_train_age, y_test_age, y_train_gender, y_test_gender)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to process dataset.')
    parser.add_argument('-p', '--path', type=str, required=True,
                        default=os.path.join(os.getenv("HOME"), "data/adience"),
                        help='Path to raw dataset file to be processed.')
    parser.add_argument('-o', '--save', type=str, required=True,
                        default=os.path.join(os.getenv("HOME"), "data/adience/adience.h5"),
                        help='Path to save the .h5 file')
    args = parser.parse_args()
    p = Process(args.path, args.save)
    p.helper()
