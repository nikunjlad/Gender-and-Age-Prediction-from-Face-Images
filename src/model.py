import torch.nn as nn
import torch.nn.functional as F


class GenderNet(nn.Module):

    def __init__(self):
        super(GenderNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16 * 16 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=29)
        self.norm4 = nn.BatchNorm2d(num_features=4)
        self.norm8 = nn.BatchNorm2d(num_features=8)
        self.norm16 = nn.BatchNorm2d(num_features=16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.norm8(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm16(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm4(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 16 * 16 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class AgeNet(nn.Module):

    def __init__(self):
        super(AgeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=7 * 7 * 384, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=12)
        self.norm1 = nn.BatchNorm2d(num_features=96)
        self.norm2 = nn.BatchNorm2d(num_features=256)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(self.pool(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(self.pool(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 384)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x