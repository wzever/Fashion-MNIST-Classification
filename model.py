import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.7))

        self.block2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        
        self.block2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.7),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.fc_1 = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc_2 = nn.Linear(512, 10)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.block1(x)
        x2_1 = self.block2_1(x)
        x2_2 = self.block2_2(x)
        x = torch.cat([x2_1, x2_2], dim=1)
        x = self.block3(x)
        self.conv_feat = x.detach().cpu()
        x = self.maxpool4(x)
        x = x.view(batchsize, -1) 
        x = self.fc_1(x)
        self.fc_feat = x.detach().cpu()
        x = self.fc_2(x)
        self.final_feat = x.detach().cpu()
        x = F.log_softmax(x, dim=1)

        return x