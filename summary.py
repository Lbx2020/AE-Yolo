#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.ghost_yolo import YoloBody

if __name__ == "__main__":
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody([[6,7,8],[3,4, 5], [0,1,2]], 1).to(device)
    summary(m, input_size=(3, 512, 512))
