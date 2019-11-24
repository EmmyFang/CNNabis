import torch.nn as nn
import numpy as np
import torch


class LeafNet(nn.Module):

    def __init__(self, hyper):

        super(LeafNet, self).__init__()

        width = hyper['width']
        ker_wid = hyper['ker_wid']
        stride = hyper['stride']
        dilation = hyper['dilation']
        padding = hyper['padding']
        pool = hyper['pool']

        channel_1 = 10
        channel_2 = 20

        self.conv1 = nn.Conv2d(3, channel_1, ker_wid)
        self.pool = nn.MaxPool2d(pool)
        self.conv2 = nn.Conv2d(channel_1, channel_2, ker_wid)

        for i in range(2):
            width = (width + padding * 2 - ((ker_wid - 1) * dilation + 1) + 1) // stride // pool
        self.in_size = channel_2 * width * width

        self.fc1 = nn.Linear(self.in_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.in_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x


    #
    # def __init__(self, hyper):
    #     super(LeafNet, self).__init__()
    #     self.width = hyper['width']
    #     ker_wid = hyper['ker_wid']
    #     stride = hyper['stride']
    #     dilation = hyper['dilation']
    #     padding = hyper['padding']
    #     pool = hyper['pool']
    #     channel_1 = 128
    #     channel_2 = 256
    #     self.conv1 = nn.Conv2d(3, channel_1, ker_wid, stride=stride, dilation=dilation, padding=padding)
    #     self.pool = nn.MaxPool2d(pool)
    #     self.conv2 = nn.Conv2d(channel_1, channel_2, ker_wid, stride=stride, dilation=dilation, padding=padding)
    #     self.bn1 = nn.BatchNorm2d(channel_1)
    #     self.bn2 = nn.BatchNorm2d(channel_2)
    #     for i in range(2):
    #         self.width = (self.width + padding * 2 - ((ker_wid - 1) * dilation + 1) + 1) // stride // pool
    #     self.fc1 = nn.Linear(channel_2 * self.width * self.width, 1000)
    #     self.fc2 = nn.Linear(1000, 64)
    #     self.fc3 = nn.Linear(64, 8)
    #
    # def forward(self, x):
    #     x = self.pool(self.bn1(torch.relu(self.conv1(x))))
    #     x = self.pool(self.bn2(torch.relu(self.conv2(x))))
    #     x = x.view(-1, 40 * self.width * self.width)
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = torch.log_softmax((self.fc3(x)), 1)
    #
    #     x = x.squeeze(1)  # Flatten to [batch_size]
    #     return x
    #
    # def __init__(self, length, channels, kn, kw, cl, hs, ly, af):
    #     super(PhoneShakeNet, self).__init__()
    #
    #     self.length = length
    #     self.final_length = kn * (self.length // 2 ** (cl + 1))
    #
    #     if af == 'ReLU':
    #         self.af = nn.ReLU
    #     elif af == 'LeakyReLU':
    #         self.af = nn.LeakyReLU
    #     else:
    #         raise ValueError('not a proper activation function')
    #
    #     self.features = nn.ModuleList([nn.Conv1d(channels, kn, kw, padding=kw//2),
    #                                    self.af(inplace=True),
    #                                    nn.MaxPool1d(2)])
    #
    #     for i in range(cl):
    #         self.features.append(nn.Conv1d(kn, kn, kw, padding=kw//2))
    #         self.features.append(self.af(inplace=True))
    #         self.features.append(nn.MaxPool1d(2))
    #
    #     self.perceive = nn.ModuleList([nn.Linear(self.final_length, hs), self.af(inplace=True)])
    #
    #     for i in range(ly - 1):
    #         self.perceive.append(nn.Linear(hs, hs))
    #         self.perceive.append(self.af(inplace=True))
    #
    #     self.perceive.append(nn.Linear(hs, 26))
    #     self.perceive.append(nn.Softmax(1))
    #
    # def forward(self, x):
    #     for i in self.features:
    #         x = i(x)
    #
    #     x = x.view(x.size(0), self.final_length)
    #
    #     for i in self.perceive:
    #         x = i(x)
    #
    #     return x
    #
