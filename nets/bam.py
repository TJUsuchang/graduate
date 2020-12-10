import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelGate_up0(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_up0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes * 2, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes * 2),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class ChannelGate_up1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_up1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes * 2, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes * 2),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class DoubleChannelGate_up(nn.Module):
    def __init__(self, in_planes):
        super(DoubleChannelGate_up, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes * 4, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes * 4),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class ChannelGate_down0(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_down0, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // 2),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class ChannelGate_down1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_down1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // 2),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class DoubleChannelGate_down(nn.Module):
    def __init__(self, in_planes):
        super(DoubleChannelGate_down, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // 4),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.fc1(a)
        chout = b
        return chout

class SimpleSpatialGate_up0(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_up0, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class SimpleSpatialGate_up1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_up1, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=1,
                                                     output_padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSimpleSpatialGate_up(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSimpleSpatialGate_up, self).__init__()
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
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 4, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2_(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class SimpleSpatialGate_down0(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_down0, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class SimpleSpatialGate_down1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_down1, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSimpleSpatialGate_down(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSimpleSpatialGate_down, self).__init__()
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
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 4, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2_(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class SimpleBAM_up0(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_up0, self).__init__()
        self.channel_att = ChannelGate_up0(in_planes)
        self.spatial_att = SimpleSpatialGate_up0(in_planes)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_planes, in_planes * 2, kernel_size=3,
                                                         stride=2, padding=1, output_padding=1,
                                                         bias=False),
                                      nn.BatchNorm2d(in_planes * 2),
                                      nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.upsample(x)
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class SimpleBAM_up1(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_up1, self).__init__()
        self.channel_att = ChannelGate_up1(in_planes)
        self.spatial_att = SimpleSpatialGate_up1(in_planes)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_planes, in_planes * 2, kernel_size=3,
                                                         stride=2, padding=1, output_padding=1,
                                                         bias=False),
                                      nn.BatchNorm2d(in_planes * 2),
                                      nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.upsample(x)
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class DoubleSimpleBAM_up(nn.Module):
    def __init__(self, in_planes):
        super(DoubleSimpleBAM_up, self).__init__()
        self.channel_att = DoubleChannelGate_up(in_planes)
        self.spatial_att = DoubleSimpleSpatialGate_up(in_planes)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_planes, in_planes * 2, kernel_size=3,
                                                         stride=2, padding=1, output_padding=1,
                                                         bias=False),
                                      nn.BatchNorm2d(in_planes * 2),
                                      nn.ReLU(inplace=True))
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(in_planes * 2, in_planes * 4, kernel_size=3,
                                                         stride=2, padding=1, output_padding=1,
                                                         bias=False),
                                      nn.BatchNorm2d(in_planes * 4),
                                      nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.upsample2(self.upsample(x))
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class SimpleBAM_down0(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_down0, self).__init__()
        self.channel_att = ChannelGate_down0(in_planes)
        self.spatial_att = SimpleSpatialGate_down0(in_planes)
        self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=3,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(in_planes // 2),
                                        nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        down = self.downsample(x)
        att = self.sigmoid(chout * shout)
        a = att * down
        b = a + down
        out = b
        return out

class SimpleBAM_down1(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_down1, self).__init__()
        self.channel_att = ChannelGate_down1(in_planes)
        self.spatial_att = SimpleSpatialGate_down1(in_planes)
        self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=3,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(in_planes // 2),
                                        nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        down = self.downsample(x)
        att = self.sigmoid(chout * shout)
        a = att * down
        b = a + down
        out = b
        return out

class DoubleSimpleBAM_down(nn.Module):
    def __init__(self, in_planes):
        super(DoubleSimpleBAM_down, self).__init__()
        self.channel_att = DoubleChannelGate_down(in_planes)
        self.spatial_att = DoubleSimpleSpatialGate_down(in_planes)
        self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=3,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(in_planes // 2),
                                        nn.ReLU(inplace=True))
        self.downsample2 = nn.Sequential(nn.Conv2d(in_planes // 2, in_planes // 4, kernel_size=3,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(in_planes // 4),
                                        nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        down = self.downsample2(self.downsample(x))
        att = self.sigmoid(chout * shout)
        a = att * down
        b = a + down
        out = b
        return out
