# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from .conv import *
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))


class Fusion(nn.Module):
    def __init__(self, inc_list):
        super().__init__()


        self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.epsilon = 1e-4


    def forward(self, x):
        # print(x[0].shape,x[1].shape)
        fusion_weight = self.relu(self.fusion_weight.clone())
        fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
        return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


# class Fusion(nn.Module):
#     def __init__(self, inc_list):
#         super().__init__()
#
#
#         self.BAFM = BAFM(inc_list)
#
#
#     def forward(self, x):
#         # print(x[0].shape,x[1].shape)
#         return self.BAFM(x)



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)



# class BAFM(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#
#         # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
#         self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])
#         # self.convs = nn.ModuleList([
#         #     nn.Sequential(
#         #         GSConv(channel, channels[0]),
#         #         EMA(channels[0])  # EMA注意力
#         #     ) for channel in channels
#         # ])
#         self.EMA = nn.ModuleList([EMA(channels[0]) for channel in channels])
#     def forward(self, xs):
#         ans = torch.ones_like(xs[0])
#         target_size = xs[0].shape[2:]
#         for i, x in enumerate(xs):
#             x = self.EMA[i](x)
#             if x.shape[-1] > target_size[-1]:
#                 x = F.avg_pool2d(x, (target_size[0], target_size[1]))
#             elif x.shape[-1] < target_size[-1]:
#                 x = F.interpolate(x, size=(target_size[0], target_size[1]),
#                                       mode='bilinear', align_corners=True)
#             ans = ans * self.convs[i](x)
#         return ans

class PreUpsample(nn.Module):
    """轻量卷积投影 + 上采样"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, target_size):
        # 语义投影
        x = self.proj(x)
        # 上采样/下采样到目标尺寸
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

class BAFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 通道统一到 channels[0]，方便后续逐元素乘
        self.EMA = nn.ModuleList([EMA(c) for c in channels])
        self.proj = nn.ModuleList([PreUpsample(c, channels[0]) for c in channels])
        self.convs = nn.ModuleList([GSConv(channels[0], channels[0]) for _ in channels])

    def forward(self, xs):
        target_size = xs[0].shape[2:]  # 以第一个特征为参考分辨率
        out = None
        for i, x in enumerate(xs):
            x = self.EMA[i](x)  # EMA 注意力
            x = self.proj[i](x, target_size)  # 轻量卷积投影 + 上采样对齐
            x = self.convs[i](x)  # GSConv
            if out is None:
                out = x
            else:
                out = out * x  # Hadamard 融合
        return out
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        t=torch.cat([x_h, x_w], dim=2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


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
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


# class ASFF(nn.Module):
#     def __init__(self, rfb=False, vis=False):
#         super(ASFF, self).__init__()
#         self.level = 0
#         self.dim = [512, 256, 128]
#         self.inter_dim = self.dim[self.level]
#         # 每个level融合前，需要先调整到一样的尺度
#         if level == 0:
#             self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
#             self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
#             self.expand = add_conv(self.inter_dim, 1024, 3, 1)
#         elif level == 1:
#             self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
#             self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
#             self.expand = add_conv(self.inter_dim, 512, 3, 1)
#         elif level == 2:
#             self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
#             self.compress_level_1 = add_conv(256, self.inter_dim, 1, 1)
#             self.expand = add_conv(self.inter_dim, 256, 3, 1)
#         compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory
#
#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#         self.vis = vis
#
#     def forward(self, x_level_0, x_level_1, x_level_2):
#         if self.level == 0:
#             level_0_resized = x_level_0
#             # print(level_0_resized.shape)
#             level_1_resized = self.stride_level_1(x_level_1)
#             # print(level_1_resized.shape)
#             level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#             # print(level_2_resized.shape)
#
#         elif self.level == 1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#             level_2_resized = self.stride_level_2(x_level_2)
#         elif self.level == 2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             # print(level_0_compressed.shape)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
#             # print(level_0_resized.shape)
#             level_1_compressed = self.compress_level_1(x_level_1)
#             level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
#             # print(level_1_resized.shape)
#             level_2_resized = x_level_2
#             # print(level_2_resized.shape)
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         # 学习的3个尺度权重
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#         # 自适应权重融合
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + level_1_resized * levels_weight[:, 1:2, :,
#                                                                                               :] + level_2_resized * levels_weight[
#                                                                                                                      :,
#                                                                                                                      2:,
#                                                                                                                      :,
#                                                                                                                      :]
#         print(f'fuse_out_reduced:{fused_out_reduced.shape}')
#         out = self.expand(fused_out_reduced)
#
#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out