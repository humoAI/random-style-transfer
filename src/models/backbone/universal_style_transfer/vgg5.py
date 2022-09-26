"""
Author 
Copyright
"""
import torch
import torch.nn as nn  
import numpy as np 

class VGG5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3,64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        # self.relu5_1 = nn.ReLU(inplace=True)

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
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x, p_indices3 = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x, p_indices4= self.pool4(x)

        x = self.conv5_1(x)
        # x = self.relu5_1(x)
        return x, [p_indices1, p_indices2, p_indices3, p_indices4]


class VGG5Inv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 1)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace=True)
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
        x = self.conv5_1(x)
        
        x = self.unpool4(x)
        x = self.relu4_4(x)
        x = self.conv4_4(x)
        x = self.relu4_3(x)
        x = self.conv4_3(x)
        x = self.relu4_2(x)
        x = self.conv4_2(x)
        x = self.relu4_1(x)
        x = self.conv4_1(x)
        
        x = self.unpool3(x)
        x = self.relu3_4(x)
        x = self.conv3_4(x)
        x = self.relu3_3(x)
        x = self.conv3_3(x)
        x = self.relu3_2(x)
        x = self.conv3_2(x)
        x = self.relu3_1(x)
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

    vgg2 = VGG5()
    vgg2Inv = VGG5Inv()
    input = torch.from_numpy(np.zeros((1,3,224,224))).float()
    x,_ = vgg2(input)
    xinv = vgg2Inv(x)
    print("x", x.shape, "xinv", xinv.shape)