import torch.nn as nn
import torch
import torch.nn.functional as F

class Channelatt0(nn.Module):
    def __init__(self, in_planes):
        super(Channelatt0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        caout = b
        return caout

class Channelatt1(nn.Module):
    def __init__(self, in_planes):
        super(Channelatt1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        caout = b
        return caout

class Channelatt2(nn.Module):
    def __init__(self, in_planes):
        super(Channelatt2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        caout = b
        return caout

class Spatialatt_up1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Spatialatt_up1, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class Spatialatt_up2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Spatialatt_up2, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSpatialatt_up(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSpatialatt_up, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2_ = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 4, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2_(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class Spatialatt_down0(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Spatialatt_down0, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class Spatialatt_down1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Spatialatt_down1, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSpatialatt_down(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSpatialatt_down, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2_ = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 4, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2_(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout
