import torch.nn as nn
import torch
import torch.nn.functional as F

class H0(nn.Module):
    def __init__(self, in_planes):
        super(H0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        hout0 = self.sigmoid(b)
        return hout0

class H1(nn.Module):
    def __init__(self, in_planes):
        super(H1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        hout1 = self.sigmoid(b)
        return hout1

class H2(nn.Module):
        def __init__(self, in_planes):
            super(H2, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
            self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes),
                                     nn.ReLU(inplace=True))
            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            a = self.avg_pool(x)
            b = self.fc1(a)
            hout2 = self.sigmoid(b)
            return hout2

class W0(nn.Module):
    def __init__(self, in_planes):
        super(W0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        wout0 = self.sigmoid(b)
        return wout0

class W1(nn.Module):
    def __init__(self, in_planes):
        super(W1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        wout1 = self.sigmoid(b)
        return wout1

class W2(nn.Module):
        def __init__(self, in_planes):
            super(W2, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
            self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes),
                                     nn.ReLU(inplace=True))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            a = self.avg_pool(x)
            b = self.fc1(a)
            wout2 = self.sigmoid(b)
            return wout2

