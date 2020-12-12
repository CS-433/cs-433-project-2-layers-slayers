#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:54:41 2020

@author: cyrilvallez
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import features

#-----------------------------------------------------------------------------

class DoubleConv(nn.Module):
   "Applies a double convolution with ReLu activation and batch normalization"
   

   def __init__(self, in_channels, out_channels):
       
       super().__init__()
        
       self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

   def forward(self, x):
       return self.double_conv(x)

#-----------------------------------------------------------------------------

class Down(nn.Module):
    """Applies a MaxPooling, then double conv, to go down the U Net"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
            )
        

    def forward(self, x):
        return self.maxpool_conv(x)
    
#-----------------------------------------------------------------------------
    
class Up(nn.Module):
    """Go up in the U Net, then double conv"""
    

    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.up = nn.ConvTranspose2d(in_channels , out_channels,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
#-----------------------------------------------------------------------------
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#-----------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.in_channels = 3*(1+len(self.filters))
        
        self.ini = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512) 
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, 2)

    def forward(self, x):
        device = x.device
        
        x = x.permute(0,2,3,1)
        x = features.cat_features(x, self.filters, contrast=False)
        x = x.permute(0,3,1,2)
        x = x.to(device)
        
        # print(x.shape)
        x1 = self.ini(x)
        #print(x1.shape)
        x2 = self.down1(x1)
        #print(x2.shape)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
        #print(x4.shape)
        x5 = self.down4(x4)
        #print(x5.shape)
        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        logits = self.out_conv(x)
        #print(logits.shape)
        
        return logits
    
    
    
    