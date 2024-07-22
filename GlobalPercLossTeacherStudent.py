from ast import Yield
import torch
import torch.nn as nn
import numpy as np
import math

class GlobalPercConfig():
  def __init__(self, start_weight=1.0, end_weight=2.0, curve_force=0,
               modules_to_hook=[], transform_normalization=None,  print_data=True):
    
    self.start_weight = start_weight
    self.end_weight = end_weight
    self.curve_force = curve_force
    
    self.modules_to_hook = modules_to_hook
    self.print_data = print_data
    self.transform_normalization = transform_normalization


class GlobalPercLossTeacherStudent(torch.nn.Module):
    def __init__(self, teacher, student, config):
        super().__init__()
        
        self.config = config
        self.activations = []

        def getActivation():
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        self.teacher = teacher.eval()
        self.student = student

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
                
                
        
        traverse_modules(self.teacher)
        traverse_modules(self.student)

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

    def custom_teacher_loss(self, S, T):
        if S.dim() > 3:
            S = S.view(S.size(0), S.size(1), -1)
            T = T.view(T.size(0), T.size(1), -1)
        
        _, C, _ = T.shape
        _, Y, _ = S.shape

        group_size = C // Y
        
        T = T.view(T.size(0), Y, group_size, -1)
        T = T.sum(dim=2) / group_size

        return (1 - nn.functional.cosine_similarity(S, T.detach(), dim=-1)).pow(2).mean()

    def forward(self, X):
        layers_loss  = self._forward_features(X)
        loss = 0

        for i in range(len(layers_loss)):
            loss_l = layers_loss[i] * self.weights[i]
            loss += loss_l

        return loss
        
    
    def _forward_features(self, X):
        X = self.normalize(X)

        self.activations = []
        self.student(X)
        X_VAL = self.activations

        self.activations = []
        with torch.no_grad():
            self.teacher(X) 
        Y_VAL = self.activations

        layers_loss = []
        for i in range(len(X_VAL)):
            S = X_VAL[i]
            T = Y_VAL[i]
            
            loss = self.custom_teacher_loss(S, T) 
            layers_loss.append(loss)
              
        return layers_loss