import warnings
import numpy as np

import torch
from torch import nn


def auto_pad(k, pad=None):
    # Pad to 'same'
    if pad is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        # np.gcd -> 最大公约数
        super(DWConv, self).__init__(c1, c2, k, s, g=np.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPP(nn.Module):
    # Spatial pyramid pooling layer
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = int(c1 / 2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(int(c_ * (len(k) + 1)), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c1 * e)
        self.cv_a = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv_b = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv1 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.bottlenecks = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.cv2 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        a = self.cv_a(x)
        yb = self.cv1(self.bottlenecks(self.cv_b(x)))
        return self.cv2(self.act(self.bn(torch.cat((a, yb), dim=1))))


class Focus(nn.Module):
    def __init__(self, c1, c2, k1=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k1, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


class UpSample(nn.Module):
    # conv > up-sample > concat > c3
    def __init__(self, c1, c3, c2, scale=2, mode='nearest', n=1, shortcut=True, g=1):
        super(UpSample, self).__init__()
        self.cv1 = Conv(c1, c2)
        self.m = nn.Sequential(*[BottleneckCSP(c3, c2, shortcut, g, e=1.0) for _ in range(n)])
        self.up_sample_ = nn.Upsample(scale_factor=scale, mode=mode)

    def forward(self, x):
        if isinstance(x, list):
            x[0] = self.up_sample_(self.cv1(x[0]))
            if len(x) > 1:
                x = torch.cat(x, dim=1)
            else:
                x = x[0]
        return self.m(x)
