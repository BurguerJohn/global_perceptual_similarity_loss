from torchvision import transforms
from GlobalPercLossDiscriminator import GlobalPercLossDiscriminator, GlobalPercConfig

import torch
import torch.nn as nn
import numpy as np


############################################
# Experimental code, using Dino as discriminator for a GAN
############################################


#Dino accept images as input, so the input will need to be [B, 3, H, W]
#Pretty much any model can be feed, just make sure the input tensor is accepted by the base model.
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

#Will transform the input before feeding it to Dino
transform = transforms.Compose([
            transforms.CenterCrop(size=(70, 70)),
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
loss = GlobalPercLossDiscriminator(model, config)
real_image = torch.rand(1, 3, 70, 70)
fake_image = torch.rand(1, 3, 70, 70)

label_real = 1.
label_fake = 0.

#it need the shape of the tensors it will be used during training to generate the discriminator layers.
loss.generate_disc(torch.rand(1, 3, 70, 70).to("cpu"))

training_parameters = loss.disc.parameters()

#Since it's using a model for the loss, you may want to disable autocast.
with torch.cuda.amp.autocast(False):
    real_loss = loss(real_image, label_real)
    fake_loss = loss(fake_image, label_fake)
    print("real_loss:", real_loss, " fake_loss:", fake_loss)


