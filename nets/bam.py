import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


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

class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1

            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)

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
