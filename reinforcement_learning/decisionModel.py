import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecision(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecision, self).__init__()
        
        # Encoder
        self.conv1 = self.conv_block(in_channels, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 128)
        self.conv5 = self.conv_block(128, 128)

        self.depthwise = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.depthwise2 = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
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

        #40x25 pixels x 128 channels [1, 128, 25, 40]
        dw1 = self.depthwise(c5)
        dw = self.depthwise2(dw1)

        #40x25 pixels x 64 channels [1, 64, 25, 40]
        dw_flat = dw.reshape(dw.size(0), -1)
        # Flattened to [1, 8000]
        logits = self.fc(dw_flat)
        # output is [1, out_channels]
        return logits
    


