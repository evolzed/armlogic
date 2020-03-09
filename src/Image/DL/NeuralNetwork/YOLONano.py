import torch
from torch import nn
from .Config import CLASS_NUM


class UpSampleLayer(nn.Module):
    """上采样，扩大特征尺寸"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class ConvolutionalLayer(nn.Module):
    """卷积层"""

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


class EPLayer(nn.Module):
    """Expansion Projection Layer"""

    def __init__(self, inChannels, outChannels, stride=1):
        super().__init__()
        self.subModule = nn.Sequential(
            ConvolutionalLayer(inChannels, outChannels, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(outChannels, outChannels, 3, stride, 1,
                               groups=outChannels),  # 深度分离卷积
            ConvolutionalLayer(outChannels, outChannels, 1)  # 逐点卷积进行通道信息融合，通道尺寸保持
        )

    def forward(self, x):
        return x + self.subModule(x)


class PEPLayer(nn.Module):
    """Projection Expansion Projection Layer"""

    def __init__(self, inChannels, outChannels, projection):
        super().__init__()
        self.subModule = nn.Sequential(
            ConvolutionalLayer(inChannels, projection, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(projection, projection, 1),  # 逐点卷积扩通道
            ConvolutionalLayer(projection, projection, 3, 1, 1,
                               groups=projection),  # 深度分离卷积
            ConvolutionalLayer(projection, outChannels, 1)  # 逐点卷积进行通道信息融合，通道尺寸保持
        )

    def forward(self, x):
        return x + self.subModule(x)


class FCALayer(nn.Module):
    """Fully Connected Attention Layer"""

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


class YOLONano(nn.Module):
    """YOLO Nano主网络"""

    def __init__(self):
        super().__init__()
        self.zoom_8 = nn.Sequential(
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
        self.zoom_16 = nn.Sequential(
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
        self.zoom_32 = nn.Sequential(
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
        self.convset_16 = nn.Sequential(
            PEPLayer(430, 325, 113),
            PEPLayer(325, 207, 99),
            ConvolutionalLayer(207, 98, 1)
        )
        self.detection_16 = nn.Sequential(
            EPLayer(98, 183),
            ConvolutionalLayer(183, 3 * (5 + CLASS_NUM), 1)
        )
        self.up_8 = nn.Sequential(
            ConvolutionalLayer(98, 47, 1),
            UpSampleLayer()
        )
        self.detection_8 = nn.Sequential(
            PEPLayer(197, 122, 58),
            PEPLayer(122, 87, 52),
            PEPLayer(87, 93, 47),
            ConvolutionalLayer(93, 3 * (5 + CLASS_NUM), 1)
        )

    def forward(self, x):
        zoom_out_8 = self.zoom_8(x)  # 降采样8倍
        zoom_out_16 = self.zoom_16(zoom_out_8)
        zoom_out_32 = self.zoom_32(zoom_out_16)

        up_out_16 = self.up_16(zoom_out_32)
        route_out_16 = torch.cat((up_out_16, zoom_out_16), 1)
        conv_out_16 = self.convset_16(route_out_16)

        up_out_8 = self.up_8(conv_out_16)
        route_out_8 = torch.cat((up_out_8, zoom_out_8), 1)

        return self.detection_32(zoom_out_32), self.detection_16(conv_out_16), self.detection_8(route_out_8)
