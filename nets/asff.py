import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from utils import utils

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.2))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class ASFF0(nn.Module):
    def __init__(self, level, vis=False):
        super(ASFF0, self).__init__()
        self.level = level
        self.dim = [16, 32, 64]
        self.inter_dim = self.dim[self.level]
        self.stride_level_1 = add_conv(32, self.inter_dim, 3, 2)
        self.stride_level_2 = add_conv(64, self.inter_dim, 3, 2)
        # self.expand = add_conv(self.inter_dim, 16, 3, 1)

        self.pre_w0 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())

        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):

        level_0_resized = x_level_0
        level_1_resized = self.stride_level_1(x_level_1)
        level_2_compressed = self.stride_level_2(x_level_2)
        level_2_resized = F.interpolate(level_2_compressed, size=(x_level_0.size(2), x_level_0.size(3)), mode='bilinear', align_corners=False)

        cat0 = torch.cat((torch.max(level_0_resized, 1)[0].unsqueeze(1), torch.mean(level_0_resized, 1).unsqueeze(1)), dim=1)
        cat1 = torch.cat((torch.max(level_1_resized, 1)[0].unsqueeze(1), torch.mean(level_1_resized, 1).unsqueeze(1)), dim=1)
        cat2 = torch.cat((torch.max(level_2_resized, 1)[0].unsqueeze(1), torch.mean(level_2_resized, 1).unsqueeze(1)), dim=1)
        level_0_weight_v = self.pre_w0(cat0)
        level_1_weight_v = self.pre_w1(cat1)
        level_2_weight_v = self.pre_w2(cat2)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(levels_weight_v, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :] + \
                            x_level_0

        # out = self.expand(fused_out_reduced)
        out = fused_out_reduced

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class ASFF1(nn.Module):
    def __init__(self, level, vis=False):
        super(ASFF1, self).__init__()
        self.level = level
        self.dim = [16, 32, 64]
        self.inter_dim = self.dim[self.level]
        self.compress_level_0 = add_conv(16, self.inter_dim, 1, 1)
        self.stride_level_2 = add_conv(64, self.inter_dim, 3, 2)
        # self.expand = add_conv(self.inter_dim, 32, 3, 1)

        self.pre_w0 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())

        self.vis = vis


    def forward(self, x_level_0, x_level_1, x_level_2):

        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized = F.interpolate(level_0_compressed, size=(x_level_1.size(2), x_level_1.size(3)), mode='bilinear', align_corners=False)
        level_1_resized =level_2_resized = self.stride_level_2(x_level_2)

        cat0 = torch.cat((torch.max(level_0_resized, 1)[0].unsqueeze(1), torch.mean(level_0_resized, 1).unsqueeze(1)), dim=1)
        cat1 = torch.cat((torch.max(level_1_resized, 1)[0].unsqueeze(1), torch.mean(level_1_resized, 1).unsqueeze(1)), dim=1)
        cat2 = torch.cat((torch.max(level_2_resized, 1)[0].unsqueeze(1), torch.mean(level_2_resized, 1).unsqueeze(1)), dim=1)
        level_0_weight_v = self.pre_w0(cat0)
        level_1_weight_v = self.pre_w1(cat1)
        level_2_weight_v = self.pre_w2(cat2)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(levels_weight_v, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :] + \
                            x_level_1

        # out = self.expand(fused_out_reduced)
        out = fused_out_reduced

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class ASFF2(nn.Module):
    def __init__(self, level, vis=False):
        super(ASFF2, self).__init__()
        self.level = level
        self.dim = [16, 32, 64]
        self.inter_dim = self.dim[self.level]

        self.compress_level_0 = add_conv(16, self.inter_dim, 1, 1)
        self.compress_level_1 = add_conv(32, self.inter_dim, 1, 1)
        # self.expand = add_conv(self.inter_dim, 64, 3, 1)

        self.pre_w0 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())
        self.pre_w2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())

        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):

        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized = F.interpolate(level_0_compressed, size=(x_level_2.size(2), x_level_2.size(3)), mode='bilinear', align_corners=False)
        level_1_compressed = self.compress_level_1(x_level_1)
        level_1_resized = F.interpolate(level_1_compressed, size=(x_level_2.size(2), x_level_2.size(3)), mode='bilinear', align_corners=False)
        level_2_resized = x_level_2

        cat0 = torch.cat((torch.max(level_0_resized, 1)[0].unsqueeze(1), torch.mean(level_0_resized, 1).unsqueeze(1)), dim=1)
        cat1 = torch.cat((torch.max(level_1_resized, 1)[0].unsqueeze(1), torch.mean(level_1_resized, 1).unsqueeze(1)), dim=1)
        cat2 = torch.cat((torch.max(level_2_resized, 1)[0].unsqueeze(1), torch.mean(level_2_resized, 1).unsqueeze(1)), dim=1)
        level_0_weight_v = self.pre_w0(cat0)
        level_1_weight_v = self.pre_w1(cat1)
        level_2_weight_v = self.pre_w2(cat2)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(levels_weight_v, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :] + \
                            x_level_2

        # out = self.expand(fused_out_reduced)
        out = fused_out_reduced

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out