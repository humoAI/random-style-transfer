import torch 
import torch.nn as nn

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if(self.norm is not None):
            x = self.norm(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes: int, init_weights: bool=True, dropout:float=0.5):
        self.num_classes = num_classes
        super().__init__()
        self.features = nn.Sequential(
            Conv3x3(3, 64),
            Conv3x3(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3(64, 128),
            Conv3x3(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3(128, 256),
            Conv3x3(256, 256),
            Conv3x3(256, 256),
            Conv3x3(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3(256, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            Conv3x3(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, self.num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    import numpy as np 
    vgg19 = VGG19(1000)
    input = torch.from_numpy(np.zeros((1,3, 224, 224))).float()
    out = vgg19(input)
    print(out.shape)
