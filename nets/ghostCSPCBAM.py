import math

import torch
import torch.nn as nn
from nets.attention import cbam_block
attention_block = [cbam_block]

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(0.1) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(0.1) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''


# ---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()

        self.out_channels = out_channels

        self.conv1 = GhostModule(in_channels, out_channels, 3)

        self.conv2 = GhostModule(out_channels // 2, out_channels // 2, 3)
        self.conv3 = GhostModule(out_channels // 2, out_channels // 2, 3)

        self.conv4 = GhostModule(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

        self.feat_att = attention_block[0](out_channels*2)


    def forward(self, x):

        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        x = torch.cat([route, x], dim=1)
        x = self.feat_att(x)
        x = self.maxpool(x)
        return x


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()

        self.conv1 = GhostModule(3, 32, kernel_size=3, stride=2)
        self.conv2 = GhostModule(32, 64, kernel_size=3, stride=2)
        self.resblock_body1 = Resblock_body(64, 64)
        self.resblock_body2 = Resblock_body(128, 128)
        self.resblock_body3 = Resblock_body(256, 256)
        self.conv3 = GhostModule(512, 512, kernel_size=3)
        self.num_features = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.resblock_body1(x)
        x2 = self.resblock_body2(x1)
        x3 = self.resblock_body3(x2)
        x3 = self.conv3(x3)
        feat1 = x1
        feat2 = x2
        feat3 = x3

        return feat1, feat2, feat3


def darknet53_tiny(pretrained, **kwargs):

    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
