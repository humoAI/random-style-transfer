"""
Author 
Copyright
"""
import torch
import torch.nn as nn  
import numpy as np 
class VGG1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1_1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(3, 64, 3, 1, 1)
        # self.relu1_1 = nn.ReLU(inplace=True)
    def forward(self, x):
        # x = self.conv1_1(x)
        x = self.conv1_2(x)
        # x = self.relu1_1(x)
        return x
class VGG1Inv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 1)
        # self.relu1_1 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1_1(x)
        return x

if __name__ == "__main__":
    # from torch.utils.serialization import load_lua
    # a = load_lua('C:/Users/Challenger/Documents/projects/kaggle/random-style-transfer/src/data/universal_style_transfer_weights/feature_invertor_conv1_1.t7')
    vgg1 = VGG1()
    vgg1Inv = VGG1Inv()
    input = torch.from_numpy(np.zeros((1,3,224,224))).float()
    x = vgg1(input)
    xinv = vgg1Inv(x)
    print("x", x.shape, "xinv", xinv.shape)
