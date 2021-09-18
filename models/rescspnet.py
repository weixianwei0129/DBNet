import torch
from torch import nn

import numpy as np
from models.common import Focus, Conv, UpSample
from models.common import DWConv, SPP, BottleneckCSP


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return int(np.ceil(x / divisor) * divisor)


class ResNetCSP(nn.Module):
    # backbone
    def __init__(self, cfg):
        super(ResNetCSP, self).__init__()
        ch = [3]  # 输入图片的通道数
        width = cfg.width_multiple
        layers = []
        for index, (m, n, args, store) in enumerate(cfg.backbone + cfg.neck):
            m = eval(m) if isinstance(m, str) else m
            if m is UpSample:
                c1 = ch[-1]
                c2 = args[1]
                if isinstance(args[0], list):
                    fetch = []
                    for ci in args[0]:
                        fetch.append(ci % len(ch))
                    c3 = sum([ch[i] for i in fetch[1:]]) + c2
                else:
                    raise ValueError("UpSample Error args", args)
                args = [c1, c3, *args[1:]]
            elif m in [DWConv, SPP, BottleneckCSP, Focus, Conv]:
                fetch = None
                c1, c2 = ch[-1], make_divisible(args[0] * width, 8)
                args = [c1, c2, *args[1:]]
            else:
                raise ValueError("Error module name!")
            if index == 0:
                ch = []
            ch.append(c2)
            if n > 1:
                m_ = nn.Sequential(*[m(*args) for _ in range(n)])
            else:
                m_ = m(*args)
            m_.store = store
            m_.fetch = fetch
            layers.append(m_)
        self.model = nn.Sequential(*layers)
        self.conv = nn.Conv2d(ch[-1], 2, 1, 1, bias=False)

    def forward(self, x):
        y = []
        for m in self.model:
            if isinstance(m.fetch, list):
                if len(m.fetch) > 1:
                    x = [x, y[m.fetch[1]]]
                else:
                    x = [x]
            x = m(x)
            y.append(x if m.store else None)
        del y
        return self.conv(x)


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from torchsummary import summary

    x = torch.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.float32))
    cfg = EasyDict(yaml.safe_load(open('../config/db_v1.0.0.yaml')))
    model = ResNetCSP(cfg.model)
    summary(model, (3, 320, 320))
    # x = model(x)
    # print(x.shape)
