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
        
        self.layers = nn.ModuleList([nn.Conv2d(3, 3, 1, 1, 0),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(3, 64, 3, 1, 0),
                                     nn.ReLU(inplace=True)])
    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

def load_weights(lua_model, vgg_pytorch_model):
    for i in range(len(lua_model)):
        try:
            vgg_pytorch_model.layers[i].weight = torch.nn.Parameter(lua_model.get(i).weight.float())
            vgg_pytorch_model.layers[i].bias = torch.nn.Parameter(lua_model.get(i).bias.float())
        except Exception as e:
            print(f"module {vgg_pytorch_model.layers[i]} doesnt have weight to load")

def save_pytorch_weights(vgg_pytorch_model, name):
    import os 
    path= os.path.join("src","data", "universal_style_transfer_weights", "pytorch")
    if(not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".pth")
    torch.save(vgg_pytorch_model.state_dict(), path)

class VGG1Inv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0)
        ])
    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

if __name__ == "__main__":
    from torch.utils.serialization import load_lua
    a = load_lua('src/data/universal_style_transfer_weights/lua/models/vgg_normalised_conv1_1.t7')
    print(a)
    print("inventor")
    aInv = load_lua('src/data/universal_style_transfer_weights/lua/decoders_noCudnn/feature_invertor_conv1_1.t7')
    print(aInv)

    vgg1 = VGG1()
    vgg1Inv = VGG1Inv()
    input = torch.from_numpy(np.zeros((1,3,224,224))).float()
    x = vgg1(input)
    xinv = vgg1Inv(x)
    print("x", x.shape, "xinv", xinv.shape)
    load_weights(a, vgg1)
    load_weights(aInv, vgg1Inv)

    save_pytorch_weights(vgg1, "vgg1")
    save_pytorch_weights(vgg1Inv, "vgg1Inv")