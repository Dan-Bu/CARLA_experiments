import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 512)

        self.depthwise = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64000, 16000),
            nn.ReLU(),
            nn.Linear(16000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    
    
    def forward(self, x):
        c1 = self.conv1(x)                                          # scale 1
        c2 = self.conv2(F.max_pool2d(c1,kernel_size=2, stride=2))   # scale 1/2
        c3 = self.conv3(F.max_pool2d(c2,kernel_size=2, stride=2))   # scale 1/4
        c4 = self.conv4(F.max_pool2d(c3,kernel_size=2, stride=2))   # scale 1/8
        c5 = self.conv5(F.max_pool2d(c4,kernel_size=2, stride=2))   # scale 1/16

        #1000 pixels x 512 channels

        dw = self.depthwise(c5)

        #1000 pixels x 64 channels

        logits = self.fc(dw)

        return logits
    


