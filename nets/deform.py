import torch.nn as nn
import torch
from nets.deform_conv import DeformConv, ModulatedDeformConv

class globalpoolatten3(nn.Module):
    def __init__(self):
        super(globalpoolatten3, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv1(cat)
        out = self.sigmoid(atten) * x
        return out

class globalpoolatten5(nn.Module):
    def __init__(self):
        super(globalpoolatten5, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv1(cat)
        out = self.sigmoid(atten) * x
        return out

class globalpoolatten7(nn.Module):
    def __init__(self):
        super(globalpoolatten7, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv1(cat)
        out = self.sigmoid(atten) * x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
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

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeformConv2d(nn.Module):
    """A single (modulated) deformable conv layer"""

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=2,
                 groups=1,
                 deformable_groups=2,
                 modulation=True,
                 double_mask=True,
                 bias=False):
        super(DeformConv2d, self).__init__()

        self.modulation = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.double_mask = double_mask

        if self.modulation:
            self.deform_conv = ModulatedDeformConv(in_channels,
                                                   out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=dilation,
                                                   dilation=dilation,
                                                   groups=groups,
                                                   deformable_groups=deformable_groups,
                                                   bias=bias)
        else:
            self.deform_conv = DeformConv(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          groups=groups,
                                          deformable_groups=deformable_groups,
                                          bias=bias)

        k = 3 if self.modulation else 2

        offset_out_channels = deformable_groups * k * kernel_size * kernel_size

        # Group-wise offset leraning when deformable_groups > 1
        self.offset_conv = nn.Conv2d(in_channels, offset_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=dilation, dilation=dilation,
                                     groups=deformable_groups, bias=True)

        # Initialize the weight for offset_conv as 0 to act like regular conv
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        if self.modulation:
            offset_mask = self.offset_conv(x)

            offset_channel = self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            offset = offset_mask[:, :offset_channel, :, :]

            mask = offset_mask[:, offset_channel:, :, :]
            mask = mask.sigmoid()  # [0, 1]

            if self.double_mask:
                mask = mask * 2  # initialize as 1 to work as regular conv

            out = self.deform_conv(x, offset, mask)

        else:
            offset = self.offset_conv(x)
            out = self.deform_conv(x, offset)

        return out


class DeformBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.ca = ChannelAttention(planes * self.expansion)
        # self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.sa(out) * out
        # out = self.ca(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeformSimpleBottleneck(nn.Module):
    """Used for cost aggregation"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,
                 mdconv_dilation=2,
                 deformable_groups=2,
                 modulation=True,
                 double_mask=True,
                 ):
        super(DeformSimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride,
                                  dilation=mdconv_dilation,
                                  deformable_groups=deformable_groups,
                                  modulation=modulation,
                                  double_mask=double_mask)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
