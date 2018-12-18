# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:10:59 2018

@author: Junsik
"""

import torch 
import torch.nn as nn

    # CNN Model (3 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential( #ConvOutput = (W-F+2*P)/S+1
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),# input channel, output channel
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))# default stride value is kernel_size = 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Linear(56000,100)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__() #input: Batch,Channel,30,500
        self.conv1 = nn.Sequential( # ConvOutput = (W-F+2*P)/S+1
                nn.Conv2d(6,32,3,1,1),
                nn.Conv2d(32,64,3,1,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # PoolingOutput = (W-F)/S+1
                ) 

        self.conv2 = nn.Sequential(
                nn.Conv2d(64,128,3,1,1),
                nn.Conv2d(128,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ) 

        self.conv3 = nn.Sequential(
                nn.Conv2d(256,256,3,1,1),
                nn.Conv2d(256,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ) 

#        self.conv4 = nn.Sequential(
#                nn.Conv2d(256,512,3,1,1),
#                nn.Conv2d(512,512,3,1,1),
#                nn.BatchNorm2d(512),
#                nn.ReLU(),
#                nn.MaxPool2d(kernel_size=2, stride=2),
#                ) 

        self.out = nn.Linear(47616, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0),-1) # Flatten the output of conv3 layer to (batch_size x data_flattened_size)
        output = self.out(x)
        return output
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.layer1 = nn.Sequential(#ConvOutput = (W-F+2*P)/S+1
            nn.Conv2d(6, 16, kernel_size=3, padding=1),# input channel, output channel
            nn.BatchNorm2d(16),
            nn.ReLU()
            )# default stride value is kernel_size = 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(56000,100)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
#####    
class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        moment = 0.7
        self.layer1 = nn.Sequential(#ConvOutput = (W-F+2*P)/S+1
            # input channel, output channel, kernel(filter_size), stride, padding
            nn.Conv2d(6, 32, kernel_size=3, stride = 2, padding=1),
#            nn.BatchNorm2d(32, momentum = moment),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
#            nn.BatchNorm2d(64, momentum = moment),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),# output has same size as imput
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, momentum = 0.5),
#            nn.BatchNorm2d(256, momentum = moment),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, momentum = moment),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256, momentum = moment),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1,1),
            nn.BatchNorm2d(512, momentum = moment),
            nn.ReLU())
        # Global Averaging
        
        self.fc = nn.Linear(512,100)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    