import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_class=4, init=True):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=init)
        self.ouput = nn.Sequential(nn.Linear(1000, 100),nn.Linear(100, num_class))

    def forward(self, x):
        x = self.model(x)
        x = self.ouput(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    conroi = torch.randn((10, 3, 300, 300))
    out = model(conroi)
    print(out.shape)

