from ast import Yield
import torch
import torch.nn as nn
import numpy as np
import math

class GlobalPercConfig():
  def __init__(self, start_weight=1.0, end_weight=2.0, curve_force=0, modules_to_hook=[], transform_normalization=None, loss_func=None, print_data=True):
    self.start_weight = start_weight
    self.end_weight = end_weight
    self.curve_force = curve_force
    
    self.modules_to_hook = modules_to_hook
    self.print_data = print_data
    self.transform_normalization = transform_normalization
    self.loss_func = loss_func


class GlobalPercLoss(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        
        self.activations = []

        def getActivation():
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        self.model = model.eval()

        count = 0

        def traverse_modules(module):
            nonlocal count
            for name, sub_module in module.named_children():                  
                traverse_modules(sub_module)

                if (len(config.modules_to_hook) == 0 and len(list(sub_module.named_children())) == 0) or isinstance(sub_module, tuple(config.modules_to_hook)):
                  if config.print_data:          
                    print("~ Hook in module:", sub_module)
                  count += 1
                  sub_module.register_forward_hook(getActivation())
                
                
        
        traverse_modules(self.model)

        if config.curve_force == 0:
            self.weights = np.linspace(config.start_weight, config.end_weight, count)
        else:
            self.weights = self.cosine_interpolation(config.start_weight, config.end_weight, config.curve_force, count)
        
        if config.print_data:
          print(f"~ Total Layers Hook: {count}")
          print(f"~ Weight for each Hook: ", self.weights)


        self.normalize = config.transform_normalization
        self.loss_func = config.loss_func
    
    def cosine_interpolation(self, start, end, strength, num_points):
        t = torch.linspace(0, 1, num_points)
        t = (1 - torch.cos(t * math.pi)) / 2  # Apply cosine interpolation
        t = t.pow(strength)  # Apply strength
        return start + (end - start) * t

    def forward(self, X, Y):

        X = self.normalize(X)
        Y = self.normalize(Y.detach())

        self.activations = []
        self.model(X)
        X_VAL = self.activations

        self.activations = []
        with torch.no_grad():
            self.model(Y) 
        Y_VAL = self.activations

        loss = 0
        for i in range(len(X_VAL)):
            A = X_VAL[i]
            B = Y_VAL[i]
            
            loss += self.loss_func(A, B) * self.weights[i]
              
        return loss