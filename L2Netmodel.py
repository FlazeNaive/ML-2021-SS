import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class L2Net(nn.Module):

    def __init__(self, in_ch, descriptor_size):

        super().__init__()

        self.descriptor_size = descriptor_size
        self.affine = False

        self.conv1 = torch.nn.Conv2d(in_ch, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-4, affine=self.affine)

        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4, affine=self.affine)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-4, affine=self.affine)

        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-4, affine=self.affine)

        self.conv5 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-4, affine=self.affine)

        self.conv6 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128, eps=1e-4, affine=self.affine)

        self.conv7 = torch.nn.Conv2d(128, descriptor_size, 8, stride=1)
        self.bn7 = nn.BatchNorm2d(descriptor_size, eps = 1e-4, affine=self.affine)
        self.lrn = torch.nn.LocalResponseNorm(size=256, alpha = 256, beta = 0.5, k = 0)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout2d(p=0.3)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        # x = self.drop(x)
        x = self.lrn(self.bn7(self.conv7(x))).squeeze()

        return x # F.normalize(x, p=2, dim=1)

    def _forward(self, xa, xp):
        return self._forward(xa), self._forward(xp)