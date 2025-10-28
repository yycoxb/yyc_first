import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from .attention import *
from .block import C3k,C3k2,Bottleneck
from .conv import *





class Bottleneck_EMA(Bottleneck):
    """Standard bottleneck With CloAttention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=..., e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        #self.attention = EfficientAttention(c2)
        self.attention = EMA(c2)
        # self.attention = LSKBlock(c2)

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        #print(f'c2:{self.c2}')
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_EMA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_EMA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EMA(self.c, self.c, shortcut, g) for _ in range(n))




##################################################original ######################################################################################



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

    
class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU())
        )
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out


class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * out * x + self.b * x


class DualDomainSelectionMechanism(nn.Module):
    # https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_Focal_Network_for_Image_Restoration_ICCV_2023_paper.pdf
    # https://github.com/c-yn/FocalNet
    # Dual-DomainSelectionMechanism
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = DSM_SpatialGate(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a * out + self.b * x


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class LogEdgeDetector(nn.Module):
    def __init__(self, in_channels, sigma=1.0):
        super(LogEdgeDetector, self).__init__()
        self.gaussian_blur = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels,
                                       bias=False)
        self.laplacian = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                                   bias=False)

        gaussian_kernel = self._create_gaussian_kernel(sigma)
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)

        self.gaussian_blur.weight.data = gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)


    def _create_gaussian_kernel(self, sigma):
        size = 5
        x = torch.arange(-size // 2 + 1., size // 2 + 1.)
        y = x[:, None]
        x0 = y0 = 0
        g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def forward(self, x):
        blurred = self.gaussian_blur(x)
        return self.laplacian(blurred)


# class EdgeEnhancer(nn.Module):
#     def __init__(self, in_dim, edge_type='log'):
#         super().__init__()
#         self.edge_type = edge_type  # avg or sobel edge detection
#         self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
#
#         #avg
#         self.pool = nn.AvgPool2d(3, stride=1, padding=1)
#
#         # Sobel filter for edge detection
#         self.sobel_x = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=in_dim)
#         self.sobel_x.weight.data.fill_(0)
#         sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
#         for i in range(in_dim):
#             self.sobel_x.weight.data[i, 0] = sobel_kernel_x
#
#         #log
#         self.log_edge_detector = LogEdgeDetector(in_dim)
#     def forward(self, x):
#         print(self.edge_type)
#         if self.edge_type == 'avg':
#             edge = self.pool(x)
#             edge = x - edge
#         elif self.edge_type == 'sobel':
#             #print(1)
#             edge = self.sobel_x(x)  # Sobel edge detection
#         elif self.edge_type == 'log':
#             #print(2)
#             edge = self.log_edge_detector(x)
#         edge = self.out_conv(edge)
#         return x + edge

class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = DualDomainSelectionMechanism(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):

        x_size = x.size()
        out = [self.local_conv(x)]

        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))


class C3k_MutilScaleEdgeInformationSelect(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect(c_, [3, 6, 9, 12]) for _ in range(n)))

class C3k2_MutilScaleEdgeInformationSelect(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationSelect(self.c, self.c, 2, shortcut, g) if c3k else MutilScaleEdgeInformationSelect(self.c, [3, 6, 9, 12]) for _ in range(n))



################################################c3k2-mseis-CBAM###########################################################################



class MutilScaleEdgeInformationSelect_CBAM(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = CBAM(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))


class C3k_MutilScaleEdgeInformationSelect_CBAM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect_CBAM(c_, [3, 6, 9, 12]) for _ in range(n)))

class C3k2_MutilScaleEdgeInformationSelect_CBAM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationSelect_CBAM(self.c, self.c, 2, shortcut, g) if c3k else MutilScaleEdgeInformationSelect_CBAM(self.c, [3, 6, 9, 12]) for _ in range(n))

################################################c3k2-mseis-EMA###########################################################################



class MutilScaleEdgeInformationSelect_EMA(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            #self.ees.append(EdgeEnhancer(inc // len(bins),'log'))
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = EMA(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        #print(f'x:{x.shape}')
        x_size = x.size()
        out = [self.local_conv(x)]
        #print(f'out{out[0].shape}')
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))


class C3k_MutilScaleEdgeInformationSelect_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect_EMA(c_, [3, 6, 9, 12]) for _ in range(n)))

class C3k2_MutilScaleEdgeInformationSelect_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationSelect_EMA(self.c, self.c, 2, shortcut, g) if c3k else MutilScaleEdgeInformationSelect_EMA(self.c, [3, 6, 9, 12]) for _ in range(n))


