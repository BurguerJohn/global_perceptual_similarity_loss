# Global Perceptual Similarity Loss
This code aims to extract the maximum amount of information from an already trained PyTorch model and use this information to help train a new model, making it simple to replace or add to the the loss function in an existing script.

## Simple example
Original Code:
```
tensor_1 = torch.rand(1, 3, 504, 504)
tensor_2 = torch.rand(1, 3, 504, 504)
l2loss = nn.functional.mse_loss(tensor_1, tensor_2)
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

#Create the configuration for the main class
config = GlobalPercConfig(start_weight=1.,
                          end_weight=1.,
						  curve_force = 1.,
                          modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
                          transform_normalization=transform,
                          print_data = True
                          )

tensor_1 = torch.rand(1, 3, 504, 504)
tensor_2 = torch.rand(1, 3, 504, 504)

#Generate the loss
loss = GlobalPercLoss(model, config)
model_loss = loss(tensor_1, tensor_2)
l2loss = nn.functional.mse_loss(tensor_1, tensor_2) * len(loss.weights)
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
*GlobalPercLoss* is working very well, but I continue testing and trying to develop more optimized codes so it can run lighter during the backward pass.


## Experimentation with diffusion models:
This is just a experimentation to input a latent space tensor instead of a RGB image to dino.
First, I took the DinoV2 model and trained a new "head layer" for it. Instead of accepting RGB images, I trained it to accept the Latent Space of Stable Diffusion 1.5.

After that, it can be used to fine-tune an existing model of Stable Diffusion 1.5, but probably will not give good results.

If you like to test training Stable Diffusion 1.5, first download the pre-trained headers for Dino here: [Link](https://drive.google.com/drive/folders/1qcSn9LFIJHeUedPXRAu5DOj5l2Ywxxcn?usp=sharing)

After that look at the *ExampleWithDinoLatentSpace.py* script.
