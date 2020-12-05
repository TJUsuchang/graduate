import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelGate_up(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_up, self).__init__()
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


class ChannelGate_down(nn.Module):
    def __init__(self, in_planes):
        super(ChannelGate_down, self).__init__()
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

class SpatialGate_up(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SpatialGate_up, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=4,
                                                     output_padding=1, dilation=4, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSpatialGate_up(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSpatialGate_up, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.ConvTranspose2d(in_planes//ratio, in_planes // ratio,
                                                     kernel_size=3, stride=2, padding=4,
                                                     output_padding=1, dilation=4, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 4, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class SpatialGate_down(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SpatialGate_down, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=4, dilation=4, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(a)
        c = self.fs3(b)
        shout = c
        return shout

class DoubleSpatialGate_down(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DoubleSpatialGate_down, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           stride=2, padding=4, dilation=4, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU(inplace=True))
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class SimpleSpatialGate_up(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_up, self).__init__()
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
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes * 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class SimpleSpatialGate_down(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SimpleSpatialGate_down, self).__init__()
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
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes // 2, kernel_size=1, bias=False)

    def forward(self, x):
        a = self.fs1(x)
        b = self.fs2(self.fs2(a))
        c = self.fs3(b)
        shout = c
        return shout

class BAM_up(nn.Module):
    def __init__(self, in_planes):
        super(BAM_up, self).__init__()
        self.channel_att = ChannelGate_up(in_planes)
        self.spatial_att = SpatialGate_up(in_planes)
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

class DoubleBAM_up(nn.Module):
    def __init__(self, in_planes):
        super(DoubleBAM_up, self).__init__()
        self.channel_att = DoubleChannelGate_up(in_planes)
        self.spatial_att = DoubleSpatialGate_up(in_planes)
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_planes, in_planes * 2, kernel_size=3,
                                                         stride=2, padding=1, output_padding=1,
                                                         bias=False),
                                      nn.BatchNorm2d(in_planes * 2),
                                      nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.upsample(self.upsample(x))
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class BAM_down(nn.Module):
    def __init__(self, in_planes):
        super(BAM_down, self).__init__()
        self.channel_att = ChannelGate_down(in_planes)
        self.spatial_att = SpatialGate_down(in_planes)
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

class DoubleBAM_down(nn.Module):
    def __init__(self, in_planes):
        super(DoubleBAM_down, self).__init__()
        self.channel_att = DoubleChannelGate_down(in_planes)
        self.spatial_att = DoubleSpatialGate_down(in_planes)
        self.downsample = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, kernel_size=3,
                                                  stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(in_planes // 2),
                                        nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.downsample(self.downsample(x))
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class SimpleBAM_up(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_up, self).__init__()
        self.channel_att = ChannelGate_up(in_planes)
        self.spatial_att = SimpleSpatialGate_up(in_planes)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        up = self.upsample(self.upsample(x))
        att = self.sigmoid(chout * shout)
        a = att * up
        b = a + up
        out = b
        return out

class SimpleBAM_down(nn.Module):
    def __init__(self, in_planes):
        super(SimpleBAM_down, self).__init__()
        self.channel_att = ChannelGate_down(in_planes)
        self.spatial_att = SimpleSpatialGate_down(in_planes)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        down = self.downsample(self.downsample(x))
        att = self.sigmoid(chout * shout)
        a = att * down
        b = a + down
        out = b
        return out
