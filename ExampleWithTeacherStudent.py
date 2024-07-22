from torchvision import transforms
from GlobalPercLossTeacherStudent import GlobalPercConfig, GlobalPercLossTeacherStudent

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, inner_dim, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, inner_dim)
        self.enc2 = self.conv_block(inner_dim, inner_dim * 2)
        self.enc3 = self.conv_block(inner_dim * 2, inner_dim * 4)
        
        # Bottleneck
        self.bottleneck = self.conv_block(inner_dim * 4, inner_dim * 8)
        
        # Decoder
        self.dec3 = self.conv_block(inner_dim * 8, inner_dim * 4)
        self.dec2 = self.conv_block(inner_dim * 4, inner_dim * 2)
        self.dec1 = self.conv_block(inner_dim * 2, inner_dim)
        
        # Final layer
        self.final_layer = nn.Conv2d(inner_dim, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        
        # Decoder
        dec3 = self.dec3(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True))
        dec3 = dec3 + enc3
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = dec2 + enc2
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = dec1 + enc1
        
        # Final layer
        return self.final_layer(dec1)

#In a real application, teacher will already be a trained model.
teacher = UNet(3, 128, 6)

#For now, all layers need to be multiple of the teacher (128 / 32 = 4)
student = UNet(3, 32, 6)

#Will transform the input before feeding it to Dino
transform = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
])

#Create the configuration for the main class
#start_weight and end_weight are a weight multiplier for each hook on the model, from the first hook to the last.
#1.0 for start_weight/end_weight will make the same weight for all hooks.
#With start_weight=1. and end_weight=2. will make the last hook have double the weight of the first one.
#curve_force will add a curvature to the weights value
#modules_to_hook = [] will make the function try to hook on all available modules possible. 

config = GlobalPercConfig(start_weight=1.,
                          end_weight=1.,
                          curve_force = 1.,
                          modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
                          transform_normalization=transform,
                          print_data = True
                          )

#Generate the loss
loss = GlobalPercLossTeacherStudent(teacher, student, config)

#Input for both teacher and student have C = 3
tensor_1 = torch.rand(1, 3, 64, 64)

#Loss for training
loss = loss(tensor_1)
print(loss)


#After training student should learn something from teacher
student(tensor_1)

