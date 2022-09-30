"""
Author 
Copyright
"""
import torch
import torch.nn as nn  
import numpy as np 
from .vgg1 import load_weights, save_pytorch_weights

class VGG3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
                                     nn.Conv2d(3, 3, 1, 1, 0),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(3, 64, 3, 1, 0),
                                     nn.ReLU(inplace=True),

                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(64, 64, 3, 1, 0),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(64, 128, 3, 1, 0),
                                     nn.ReLU(inplace=True),

                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3, 1, 0),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),

                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 256, 3, 1, 0),
                                     nn.ReLU(inplace=True),
                                     ])
    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

class VGG3Inv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0)
        ])
    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

if __name__ == "__main__":
    from torch.utils.serialization import load_lua
    weight_folder = "src/data/universal_style_transfer_weights/lua/"
    a = load_lua(weight_folder+ 'models/vgg_normalised_conv3_1.t7')
    aInv = load_lua(weight_folder+ 'decoders_noCudnn/feature_invertor_conv3_1.t7')
    

    vgg2 = VGG3()    
    vgg2Inv = VGG3Inv()

    load_weights(a, vgg2)
    load_weights(aInv, vgg2Inv)

    input = torch.from_numpy(np.zeros((1,3,224,224))).float()
    x = vgg2(input)
    print("x", x.shape)

    xinv = vgg2Inv(x)
    print("xinv", xinv.shape)

    # print(a)
    # print("inverse")
    # print(aInv)

    save_pytorch_weights(vgg2, "vgg3")
    save_pytorch_weights(vgg2Inv, "vgg3Inv")