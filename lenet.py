from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

class LetNet_Decomposition(nn.Module):
    def __init__(self):
        super(LetNet_Decomposition, self).__init__()
        self.conv1_sigma = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2_sigma = nn.Conv2d(6, 16, 5)
        self.fc1_sigma = nn.Linear(400, 120)
        self.fc2_sigma = nn.Linear(120, 84)
        self.fc3_sigma = nn.Linear(84, 10)

        self.conv1_gamma = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2_gamma = nn.Conv2d(6, 16, 5)
        self.fc1_gamma = nn.Linear(400, 120)
        self.fc2_gamma = nn.Linear(120, 84)
        self.fc3_gamma = nn.Linear(84, 10)

    def forward(self, x):

        x_sigma = self.conv1_sigma(x)
        x_gamma = self.conv1_gamma(x)
        x = x_sigma + x_gamma
        x = F.max_pool2d(F.relu(x), 2)

        x_sigma = self.conv2_sigma(x)
        x_gamma = self.conv2_gamma(x)
        x = x_sigma + x_gamma
        x = F.max_pool2d(F.relu(x), 2)

        x = x.view(x.size(0), -1)

        x_sigma = self.fc1_sigma(x)
        x_gamma = self.fc1_gamma(x)
        x = x_sigma + x_gamma
        x = F.relu(x)

        x_sigma = self.fc2_sigma(x)
        x_gamma = self.fc2_gamma(x)
        x = x_sigma + x_gamma
        x = F.relu(x)

        x_sigma = self.fc3_sigma(x)
        x_gamma = self.fc3_gamma(x)
        x = x_sigma + x_gamma

        return x

    def predict(self, x):

        x = self.conv1_sigma(x)
        x = F.max_pool2d(F.relu(x), 2)

        x = self.conv2_sigma(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(x.size(0), -1)

        x = self.fc1_sigma(x)
        x = F.relu(x)

        x = self.fc2_sigma(x)
        x = F.relu(x)

        x = self.fc3_sigma(x)

        return x



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
       
        return out

