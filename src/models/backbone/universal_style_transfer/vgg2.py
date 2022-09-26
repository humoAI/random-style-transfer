"""
Author 
Copyright
"""
import torch
import torch.nn as nn  
import numpy as np 

class VGG2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3,64, 1, 1, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, 0, 0, return_indices=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x, p_indices1 = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        
        return x, [p_indices1]

if __name__ == "__main__":
    from torch.utils.serialization import load_lua
    weight_folder = "C:/Users/Challenger/Documents/projects/kaggle/random-style-transfer/src/data/universal_style_transfer_weights/lua/"
    a = load_lua(weight_folder+ 'feature_invertor_conv1_1.t7')