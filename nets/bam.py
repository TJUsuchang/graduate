import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((24, 48))
        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
                                 nn.BatchNorm2d(in_planes),
                                 nn.ReLU())

    def forward(self, x):
        if x.shape[1] == 64:
            if x.shape[3] == 192:
                chout = F.interpolate(self.fc2(self.fc1(self.avg_pool(x))), size=[96, 192],
                                      mode="bilinear", align_corners=False)
            elif x.shape[3] == 320:
                chout = F.interpolate(self.fc2(self.fc1(self.avg_pool(x))), size=[192, 320],
                                      mode="bilinear", align_corners=False)
        elif x.shape[1] == 32:
            if x.shape[3] == 96:
                chout = F.interpolate(self.fc2(self.fc1(self.avg_pool(x))), size=[48, 96],
                                      mode="bilinear", align_corners=False)
            elif x.shape[3] == 160:
                chout = F.interpolate(self.fc2(self.fc1(self.avg_pool(x))), size=[96, 160],
                                      mode="bilinear", align_corners=False)
        elif x.shape[1] == 16:
            if x.shape[3] == 48:
                chout = self.fc2(self.fc1(self.avg_pool(x)))
            elif x.shape[3] ==80:
                chout = F.interpolate(self.fc2(self.fc1(self.avg_pool(x))), size=[48, 80],
                                      mode="bilinear", align_corners=False)
        return chout


class SpatialGate(nn.Module):
    def __init__(self, in_planes, ratio=16, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.fs1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU())
        self.fs2 = nn.Sequential(nn.Conv2d(in_planes//ratio, in_planes // ratio, kernel_size=3,
                                           padding=dilation_val, dilation=dilation_val),
                                 nn.BatchNorm2d(in_planes // ratio),
                                 nn.ReLU())
        self.fs3 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1)

    def forward(self, x):
        shout = self.fs3(self.fs2(self.fs1(x))) # [C, H, W]
        return shout

class BAM(nn.Module):
    def __init__(self, in_planes):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(in_planes)
        self.spatial_att = SpatialGate(in_planes)
    def forward(self,x):
        chout = self.channel_att(x)
        shout = self.spatial_att(x)
        att = 1 + F.sigmoid(chout * shout)
        return att * x
