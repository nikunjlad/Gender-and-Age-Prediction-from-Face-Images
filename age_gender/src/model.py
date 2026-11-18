import torch.nn as nn
import torch.nn.functional as F


class GenderNet(nn.Module):

    def __init__(self):
        super(GenderNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=384 * 6 * 6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=2)
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm(self.pool(x))
        x = F.relu(self.conv2(x))
        x = self.norm(self.pool(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 384 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x


class AgeNet(nn.Module):

    def __init__(self):
        super(AgeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=384 * 6 * 6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=12)
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm(self.pool(x))
        x = F.relu(self.conv2(x))
        x = self.norm(self.pool(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 384 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x