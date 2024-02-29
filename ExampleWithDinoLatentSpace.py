from torchvision import transforms
from NormalizedLoss import NormalizedTensorLoss
from GlobalPercLoss import GlobalPercConfig, GlobalPercLoss


import torch
import torch.nn as nn
import numpy as np


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_in, n_out, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_in, n_out, 3, 1, 1))
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + x)

class DinoLatentHeader(nn.Module):
    def __init__(self):
        super().__init__()

        self.train_layer = nn.Sequential(
            nn.Conv2d(4, 384, 3, 1, 1),
            nn.ReLU(),
            Block(384, 384),
            Block(384, 384),
            Block(384, 384),
            nn.Conv2d(384, 384, 2, 2),
            Block(384, 384),
            Block(384, 384),
            Block(384, 384),
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
        )


    def forward(self, x):
        return self.train_layer(x)

def DinoWithLatentHeader(header_path, device="cpu"):
  model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
  head = DinoLatentHeader()
  head.load_state_dict(torch.load(header_path, map_location=device))
  model.patch_embed.proj = head
  model.patch_size = 2
  return model
  
  
model = DinoWithLatentHeader("/path/to/model/file/dino_header_rl.pkl")

#Will transform the input before feeding it to Dino
#Dino require a H,W multiple of 14, since we are using a latent of 64,64, we will add 10 pad to each side to make it 84,84
transform = transforms.Compose([
                    transforms.Pad(10, fill=0, padding_mode='constant')
            ])

#Create a custom normalized function class
#loss_func can also be nn.L1Loss(), nn.MSELoss(), or any similar, if you don't want to normalize the tensors.
loss_func = NormalizedTensorLoss("l2")


#Create the configuration for the main class
#start_weight and end_weight are a multiplier for each hook on the model, from the first hook to the last.
#1.0 for start_weight/end_weight will make the same weight for all hooks.
#With start_weight=1. and end_weight=2. will make the last hook have double the weight of the first one.
#modules_to_hook = [] will make the function try to hook on all available modules possible. 

config = GlobalPercConfig(start_weight=1.,
                          end_weight=10.,
                          curve_force = 3,
                          modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
                          transform_normalization=transform,
                          loss_func=loss_func,
                          print_data = True
                          )

#Generate the loss class
loss = GlobalPercLoss(model, config)
#Latent Tensors have 4 channels so input is [B, 4, H, W]
tensor_1 = torch.rand(1, 4, 64, 64)
tensor_2 = torch.rand(1, 4, 64, 64)

#Unlike nn.L1Loss(), nn.MSELoss(), the tensors will be feed to a model, so if you are using autocast, it may cause problems.
with torch.cuda.amp.autocast(False):
    print("Loss:", loss(tensor_1, tensor_2))

