# Global Perceptual Similarity Loss
This code aims to extract the maximum amount of information from an already trained PyTorch model and use this information to help train a new model, simply by replacing the loss function in an existing script.

## Simple example
Original Code:
```
loss = nn.L1Loss()
tensor_1 = torch.rand(1, 3, 504, 504)
tensor_2 = torch.rand(1, 3, 504, 504)
loss(tensor_1, tensor_2)
```
New Code:
```
#Using Dino as example (can be any model). Dino accept images as input, so the input will need to be [B, 3, H, W]
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

#Will transform the input before feeding it to Dino
transform = transforms.Compose([
            transforms.CenterCrop(size=(504, 504)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ])

#Create a custom normalized function class
loss_func = NormalizedTensorLoss("l2")

#Create the configuration for the main class
config = GlobalPercConfig(start_weight=1.,
                          end_weight=2.,
						  curve_force = 3,
                          modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
                          transform_normalization=transform,
                          loss_func=loss_func,
                          print_data = True
                          )

#Generate the loss
loss = GlobalPercLoss(model, config)
tensor_1 = torch.rand(1, 3, 504, 504)
tensor_2 = torch.rand(1, 3, 504, 504)
loss(tensor_1, tensor_2)
```

## What this actually do:
The code creates a hook for all classes selected by "modules_to_hook" and stores these tensors during the "forward pass"; all these tensors are used to calculate the final loss.

You can open *GlobalPercLoss.py* to see how it handle the hooks.

Since it do a lot more of complex calculations than L1Loss and MSELoss, you can expect to use more memory and be slower to compute.

## How to use:
Open *ExampleWithDino.py* to see how to use the script. The example use DINOv2 by meta, but you can use any pre-trained model you like.

If you like to test training Stable Diffusion 1.5:
- first download the pre-trained headers for Dino here: [Link](https://drive.google.com/drive/folders/1qcSn9LFIJHeUedPXRAu5DOj5l2Ywxxcn?usp=sharing)
- After that look at the *ExampleWithDinoLatentSpace.py* script.


## Weights parameters in GlobalPercConfig:
The variables **start_weight**, **end_weight** and **curve_force** control the weight of each hook of the loss model.
- **start_weight** Is the value of the first hook of the model
- **end_weight** Is the value of the last hook of the model
- **curve_force** Create a interpolation curve between **start_weight** and **end_weight**. Force: 1 it disable the curve.
### Weight visualization:
![Curve 1](/img/curves/1.png)
![Curve 2](/img/curves/2.png)
![Curve 3](/img/curves/3.png)



## Current state of the project:
*GlobalPercLoss* is working very well, but I continue testing and trying to develop more optimized codes so it can run lighter during the backward pass. I am still testing various ways to normalize the tensors to improve the results for *NormalizedLoss*, which will likely undergo many changes.


## Experimentation with diffusion models:
I began the development of this project to be used in conjunction with a personal project of mine, which uses RGB images as input.

I started to wonder if this technique would work with diffusion models, so I conducted some quick tests and obtained some interesting results.

First, I took the DinoV2 model and trained a new "head layer" for it. Instead of accepting RGB images, I trained it to accept the Latent Space of Stable Diffusion 1.5.

After that, all I needed to do was fine-tune an existing model of Stable Diffusion 1.5.

If you like to test training Stable Diffusion 1.5, first download the pre-trained headers for Dino here: [Link](https://drive.google.com/drive/folders/1qcSn9LFIJHeUedPXRAu5DOj5l2Ywxxcn?usp=sharing)

After that look at the *ExampleWithDinoLatentSpace.py* script.

### Example Images:

### What can be improved for Diffusion Loss:
- The dino header was trained with [Tiny AutoEncoder for Stable Diffusion](https://github.com/madebyollin/taesd). It probably would be better trained with the default encoder for SD 1.5
- The Dino Header is a bunch of conv2D throw together, a proper header may be able to improve the model.
- I have zero experiencie training SD models, so someone with more experience may be able to get better results. 
- Instead of DINO, use a model like [Segment Anything](https://segment-anything.com/) to feed the prompts+latents to the loss function.