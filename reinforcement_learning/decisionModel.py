import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDecision(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecision, self).__init__()
        
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
    


class ConvDecisionV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecisionV2, self).__init__()
        
        # Encoder
        self.conv1 = self.conv_block_5x5(in_channels, 24)
        self.conv2 = self.conv_block_5x5(24, 36)
        self.conv3 = self.conv_block_5x5(36, 48)
        self.conv4 = self.conv_block_3x3(48, 64)
        self.conv5 = self.conv_block_3x3(64, 64)
        self.conv6 = self.conv_block_3x3(64, 64, 2)
        self.conv7 = self.conv_block_3x3(64, 64, 0)
        self.conv8 = self.conv_block_3x3(64, 64)

        self.fc = nn.Sequential(
            nn.Linear(960, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, out_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def conv_block_3x3(self, in_channels, out_channels, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def conv_block_5x5(self, in_channels, out_channels, padding=2, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    
    def forward(self, x):
        c1 = self.conv1(x)                                          # scale 1 (640x400)
        c2 = self.conv2(F.max_pool2d(c1,kernel_size=2, stride=2))   # scale 1/2 (320x200)
        c3 = self.conv3(F.max_pool2d(c2,kernel_size=2, stride=2))   # scale 1/4 (160x100)
        c4 = self.conv4(F.max_pool2d(c3,kernel_size=2, stride=2))   # scale 1/8 (80x50)
        c5 = self.conv5(F.max_pool2d(c4,kernel_size=2, stride=2))   # scale 1/16 (40x25)
        c6 = self.conv6(F.max_pool2d(c5,kernel_size=2, stride=2))   # KS=3, S=2, P=2 -> 21x14
        c7 = self.conv7(F.max_pool2d(c6,kernel_size=2, stride=2))   # KS=3, S=2, P=0 -> 10x6
        c8 = self.conv8(F.max_pool2d(c7,kernel_size=2, stride=2, padding=1)) # KS=3, S=2, P=1 -> 5x3

        #40x25 pixels x 64 channels [1, 64, 5, 3]
        c8_flat = c8.reshape(c8.size(0), -1)
        # Flattened to [1, 960]
        logits = self.fc(c8_flat)
        # output is [1, out_channels]
        return logits
    