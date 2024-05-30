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
        
        # Decoder
        self.upsampler = self.deconv_block(512, 512)
        self.deconv4 = self.deconv_block(1024, 256)
        self.deconv3 = self.deconv_block(512, 128)
        self.deconv2 = self.deconv_block(256, 64)

        self.outconv = nn.Conv2d(128, out_channels, kernel_size=1)
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
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    
    def forward(self, x):
        c1 = self.conv1(x)                                          # scale 1
        c2 = self.conv2(F.max_pool2d(c1,kernel_size=2, stride=2))   # scale 1/2
        c3 = self.conv3(F.max_pool2d(c2,kernel_size=2, stride=2))   # scale 1/4
        c4 = self.conv4(F.max_pool2d(c3,kernel_size=2, stride=2))   # scale 1/8
        c5 = self.conv5(F.max_pool2d(c4,kernel_size=2, stride=2))   # scale 1/16

        us4 = self.upsampler(c5)
        merge4 = torch.cat([us4, c4], dim=1)
        dc4 = self.deconv4(merge4)                                  # scale 1/8
        merge3 = torch.cat([dc4, c3], dim=1)
        dc3 = self.deconv3(merge3)                                  # scale 1/4
        merge2 = torch.cat([dc3, c2], dim=1)
        dc2 = self.deconv2(merge2)                                  # scale 1/2
        merge1 = torch.cat([dc2, c1], dim=1)
        out = self.outconv(merge1)                                  # scale 1

        out = self.softmax(out)

        return out
    


