"""
Author 
Copyright
"""
import torch
import torch.nn as nn  
import numpy as np 

class VGG3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3,64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2,  return_indices=True)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        # self.relu3_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x, p_indices1 = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x, p_indices2 = self.pool2(x)

        x = self.conv3_1(x)
        # x = self.relu3_1(x)
        return x, [p_indices1, p_indices2]

class VGG3Inv(nn.Module):
    def __init__(self):
        super().__init__()

        # self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 1)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 1)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv3_1(x)
        x = self.unpool2(x)
        x = self.relu2_2(x)
        x = self.conv2_2(x)
        x = self.relu2_1(x)
        x = self.conv2_1(x)
        x = self.unpool1(x)
        x = self.relu1_2(x)        
        x = self.conv1_2(x)
        x = self.relu1_1(x)
        x = self.conv1_1(x)
        return x


if __name__ == "__main__":
    from torch.utils.serialization import load_lua
    # weight_folder = "C:/Users/Challenger/Documents/projects/kaggle/random-style-transfer/src/data/universal_style_transfer_weights/lua/"
    # a = load_lua(weight_folder+ 'feature_invertor_conv1_1.t7')

    vgg2 = VGG3()
    vgg2Inv = VGG3Inv()
    input = torch.from_numpy(np.zeros((1,3,224,224))).float()
    x,_ = vgg2(input)
    xinv = vgg2Inv(x)
    print("x", x.shape, "xinv", xinv.shape)