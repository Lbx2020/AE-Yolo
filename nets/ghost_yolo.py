import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from nets.ghostCSPCBAM import darknet53_tiny
from nets.attention import cbam_block

attention_block = [cbam_block]
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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


# ---------------------------------------------------#
#   卷积 + 上采样
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            GhostModule(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        GhostModule(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        self.epsilon = 1e-4

        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = GhostModule(512, 256, 1)
        self.p4_upsample = Upsample(256, 128)
        self.conv_for_P4 = GhostModule(256, 128, 1)

        self.conv4_up = GhostModule(256, 128, 3)

        self.p3_upsample = Upsample(128, 64)
        self.conv_for_P3 = GhostModule(128, 64, 1)

        self.conv3_up = GhostModule(128, 64, 3)
        self.yolo_headP3 = yolo_head([128, len(anchors_mask[2]) * (5 + num_classes)], 64)

        self.p4_downsample = GhostModule(64, 128, 3, stride=2)

        self.conv4_down = GhostModule(384, 128, 3)
        self.yolo_headP4 = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)], 128)

        self.p5_downsample = GhostModule(128, 256, 3, stride=2)

        self.conv5_down = GhostModule(512, 256, 3)
        self.yolo_headP5 = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)], 256)


        self.swish = Swish()

        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

    def forward(self, x):

        feat1, feat2, feat3 = self.backbone(x)

        p3_in = self.conv_for_P3(feat1)
        p4_in_1 = self.conv_for_P4(feat2)
        p4_in_2 = self.conv_for_P4(feat2)
        p5_in = self.conv_for_P5(feat3)

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_td = self.conv4_up(self.swish(torch.cat([weight[0] * p4_in_1, weight[1] * self.p4_upsample(p5_in)], axis=1)))


        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(torch.cat([weight[0] * p3_in, weight[1] * self.p3_upsample(p4_td)], axis=1)))


        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(
                torch.cat([weight[0] * p4_in_2, weight[1] * p4_td, weight[2] * self.p4_downsample(p3_out)], axis=1)))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            self.swish(torch.cat([weight[0] * p5_in, weight[1] * self.p5_downsample(p4_out)], axis=1)))


        out2 = self.yolo_headP3(p3_out)
        out1 = self.yolo_headP4(p4_out)
        out0 = self.yolo_headP5(p5_out)


        return out0, out1, out2



