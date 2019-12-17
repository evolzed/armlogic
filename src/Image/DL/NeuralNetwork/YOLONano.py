import torch
from torch import nn
from .Config import CLASS_NUM

'''上采样，扩大特征尺寸'''


class UpSampleLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


'''卷积层'''


class ConvolutionalLayer(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride=1, padding=0, groups=1, normalize=True,
                 bias=False):
        super().__init__()
        self.normShape = []
        self.subModule = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, groups=groups, bias=bias),
            nn.LayerNorm(self.normShape),
            nn.ReLU6(True)
        ) if normalize else nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, groups=groups, bias=bias),
            nn.ReLU6(True)
        )

    def forward(self, x):
        self.normShape = x.shape[2:]
        return self.subModule(x)


'''Expansion Projection Layer'''


class EPLayer(nn.Module):
    def __init__(self, inChannels, outChannels, stride=1):
        super().__init__()
        self.subModule = nn.Sequential(
            ConvolutionalLayer(inChannels, outChannels, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(outChannels, outChannels, 3, stride, 1,
                               groups=outChannels),
            # 深度分离卷积
            ConvolutionalLayer(outChannels, outChannels, 1)  # 逐点卷积进行通道信息融合，通道尺寸保持
        )

    def forward(self, x):
        return self.subModule(x)


'''Projection Expansion Projection Layer'''


class PEPLayer(nn.Module):
    def __init__(self, inChannels, outChannels, expand_ratio):
        super().__init__()
        self.subModule = nn.Sequential(
            ConvolutionalLayer(inChannels, expand_ratio, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(expand_ratio, expand_ratio, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(expand_ratio, expand_ratio, 3, 1, 1,
                               groups=expand_ratio),  # 深度分离卷积
            ConvolutionalLayer(expand_ratio, outChannels, 1)  # 逐点卷积进行通道信息融合，通道尺寸保持
        )

    def forward(self, x):
        return self.subModule(x)


'''Fully Connected Attention Layer'''


class FCALayer(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return x * out.expand_as(x)


'''YOLO Nano主网络'''


class YOLONano(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk_8 = nn.Sequential(
            ConvolutionalLayer(3, 12, 3, 1),
            ConvolutionalLayer(12, 24, 3, 2, 1),
            PEPLayer(24, 24, 7),
            EPLayer(24, 70, 2),
            PEPLayer(70, 70, 25),
            PEPLayer(70, 70, 24),
            EPLayer(70, 150, 2),
            PEPLayer(150, 150, 56),
            ConvolutionalLayer(150, 150, 1),
            FCALayer(150, 8),
            PEPLayer(150, 150, 73),
            PEPLayer(150, 150, 71),
            PEPLayer(150, 150, 75)
        )
        self.trunk_16 = nn.Sequential(
            EPLayer(150, 325, 2),
            PEPLayer(325, 325, 132),
            PEPLayer(325, 325, 124),
            PEPLayer(325, 325, 141),
            PEPLayer(325, 325, 140),
            PEPLayer(325, 325, 137),
            PEPLayer(325, 325, 135),
            PEPLayer(325, 325, 133),
            PEPLayer(325, 325, 140)
        )
        self.trunk_32 = nn.Sequential(
            EPLayer(325, 545, 2),
            PEPLayer(545, 545, 276),
            ConvolutionalLayer(545, 230, 1),
            EPLayer(230, 489),
            PEPLayer(489, 469, 213),
            ConvolutionalLayer(469, 189, 1)
        )
        self.detection_32 = nn.Sequential(
            EPLayer(189, 462),
            ConvolutionalLayer(462, 3 * (5 + CLASS_NUM), 1)
        )
        self.up_16 = nn.Sequential(
            ConvolutionalLayer(189, 105, 1),
            UpSampleLayer()
        )
        self.detection_16 = nn.Sequential(
            PEPLayer(294, 325, 113)
        )
        self.up_8 = nn.Sequential(

        )
        self.detection_8 = nn.Sequential(

        )

    def forward(self, x):
        pass
