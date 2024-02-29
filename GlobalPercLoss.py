from ast import Yield
import torch
import torch.nn as nn
import numpy as np
import math

class GlobalPercConfig():
  def __init__(self, start_weight=1.0, end_weight=2.0, curve_force=0,
               modules_to_hook=[], transform_normalization=None,
               loss_func=None, print_data=True,
               dynamic_normalization=False):
    
    self.start_weight = start_weight
    self.end_weight = end_weight
    self.curve_force = curve_force
    
    self.modules_to_hook = modules_to_hook
    self.print_data = print_data
    self.transform_normalization = transform_normalization
    self.loss_func = loss_func
    self.dynamic_normalization = dynamic_normalization


class GlobalPercLoss(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        
        self.config = config
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
        elif config.start_weight <= config.end_weight:
            self.weights = self.ease_in_curve(config.start_weight, config.end_weight, config.curve_force, count)
        else:
            self.weights = self.ease_out_curve(config.start_weight, config.end_weight, config.curve_force, count)
        if config.print_data:
          print(f"~ Total Layers Hook: {count}")
          print(f"~ Weight for each Hook: ", self.weights)


        self.normalize = config.transform_normalization
        self.loss_func = config.loss_func
        self.dynamic_norm_weights = None
        
    def ease_in_curve(self, start_value, end_value, curve_strength, qtd_points):
        # Generate a tensor of points from 0 to 1
        points = torch.linspace(0, 1, qtd_points)
        # Apply the ease-in curve (acceleration)
        eased_points = points ** curve_strength
        # Scale and offset the points to the desired range
        return start_value + (end_value - start_value) * eased_points

    def ease_out_curve(self, start_value, end_value, curve_strength, qtd_points):
        # Generate a tensor of points from 0 to 1
        points = torch.linspace(0, 1, qtd_points)
        # Apply the ease-out curve (deceleration)
        eased_points = 1 - (1 - points) ** curve_strength
        # Scale and offset the points to the desired range
        return start_value + (end_value - start_value) * eased_points

    def CreateDynamicWeights(self, tensor):
        generator = torch.Generator(tensor.device)
        generator.manual_seed(7)
        qtd_samples = 10

        dynamic_norm_weights = None
        for i in range(qtd_samples):
            t1 = torch.randn(tensor.shape, generator=generator)
            t2 = torch.randn(tensor.shape, generator=generator)
            with torch.no_grad():
                layers_loss = self._forward_features(t1, t2)
            if dynamic_norm_weights == None:
                dynamic_norm_weights = [0] * len(layers_loss)

            for ii in range(len(layers_loss)):
                dynamic_norm_weights[ii] += layers_loss[ii]
        
        for i in range(len(dynamic_norm_weights)):
            dynamic_norm_weights[i] /= qtd_samples
        
        return dynamic_norm_weights


       

    def forward(self, X, Y):
        if self.config.dynamic_normalization and self.dynamic_norm_weights == None:
            self.dynamic_norm_weights = self.CreateDynamicWeights(X)

        layers_loss  = self._forward_features(X, Y)
        loss = 0
        
        for i in range(len(layers_loss)):
            loss += layers_loss[i] * self.weights[i]
            if self.config.dynamic_normalization:
                loss *= self.dynamic_norm_weights[i]

        return loss
        
    
    def _forward_features(self, X, Y):
        X = self.normalize(X)
        Y = self.normalize(Y.detach())

        self.activations = []
        self.model(X)
        X_VAL = self.activations

        self.activations = []
        with torch.no_grad():
            self.model(Y) 
        Y_VAL = self.activations

        layers_loss = []
        for i in range(len(X_VAL)):
            A = X_VAL[i]
            B = Y_VAL[i]
            
            loss = self.loss_func(A, B) 
            layers_loss.append(loss)
              
        return layers_loss