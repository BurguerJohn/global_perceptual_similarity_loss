from torchvision import transforms
from NormalizedLoss import NormalizedTensorLoss
from GlobalPercLoss import GlobalPercConfig, GlobalPercLoss

import torch
import torch.nn as nn
import numpy as np


#Dino accept images as input, so the input will need to be [B, 3, H, W]
#Pretty much any model can be feed, just make sure the input tensor is accepted by the base model.
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

#Will transform the input before feeding it to Dino
transform = transforms.Compose([
            transforms.CenterCrop(size=(504, 504)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ])

#Create a custom normalized function class
#loss_func can also be nn.L1Loss(), nn.MSELoss(), or any similar, if you don't want to normalize the tensors.
loss_func = NormalizedTensorLoss("l2")


#Create the configuration for the main class
#start_weight and end_weight are a multiplier for each hook on the model, from the first hook to the last.
#1.0 for start_weight/end_weight will make the same weight for all hooks.
#With start_weight=1. and end_weight=2. will make the last hook have double the weight of the first one.
#In my tests, managed to get good results even with end_weight=1600
#modules_to_hook = [] will make the function try to hook on all available modules possible. 
config = GlobalPercConfig(start_weight=1.,
                          end_weight=2.,
                          modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
                          transform_normalization=transform,
                          loss_func=loss_func,
                          print_data = True
                          )

#Generate the loss
loss = GlobalPercLoss(model, config)
tensor_1 = torch.rand(1, 3, 504, 504)
tensor_2 = torch.rand(1, 3, 504, 504)

#Unlike nn.L1Loss(), nn.MSELoss(), the tensors will be feed to a model, so if you are using autocast, it may cause problems.
with torch.cuda.amp.autocast(False):
    print("Loss:", loss(tensor_1, tensor_2))

