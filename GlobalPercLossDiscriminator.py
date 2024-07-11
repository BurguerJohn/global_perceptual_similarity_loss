from ast import Yield
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class Discrimator(nn.Module): 
    def __init__(self, embed_dim, num_heads):
        super(Discrimator, self).__init__()
        self.att = SelfAttention(embed_dim, num_heads)
        self.out = nn.Linear(embed_dim, 1)
    def forward(self, x):
        x = self.att(x)
        x = self.out(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        num_heads = self.closest_divisor(embed_dim, num_heads)
        #print(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def closest_divisor(self, X, Y):
        if X % Y == 0:
            return Y
        for i in range(Y-1, 0, -1):
            if X % i == 0:
                return i
        return 1
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Apply linear transformations to get Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Multiply attention weights with values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply final linear transformation
        output = self.out(attn_output)
        
        return output

class GlobalPercConfig():
  def __init__(self, start_weight=1.0, end_weight=2.0, curve_force=0,
               modules_to_hook=[], transform_normalization=None,  print_data=True):
    
    self.start_weight = start_weight
    self.end_weight = end_weight
    self.curve_force = curve_force
    
    self.modules_to_hook = modules_to_hook
    self.print_data = print_data
    self.transform_normalization = transform_normalization


class GlobalPercLossDiscriminator(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        
        self.config = config
        self.activations = []

        def getActivation():
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        self.model = model.eval()
        self.disc = None

        self.bceLogLoss = nn.BCEWithLogitsLoss()


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

    def cosine_loss(self, A, B):
        if A.dim() > 3:
            A = A.view(A.size(0), A.size(1), -1)
            B = B.view(B.size(0), B.size(1), -1)
            
        return (1 - nn.functional.cosine_similarity(A, B.detach(), dim=-1)).pow(2).mean()

    def generate_disc(self, X):
        self.disc = nn.ModuleList()

        X = self.normalize(X)
        self.activations = []
        self.model(X)
        X_VAL = self.activations
        for i in range(len(X_VAL)):
            A = X_VAL[i]
            A = A.view(A.size(0), A.size(1), -1)
            inn = A.shape[-1]
            lin = Discrimator(inn, 1)
            self.disc.append(lin)


    def forward(self, X, label):
        layers_loss  = self._forward_features(X, label)
        loss = 0

        for i in range(len(layers_loss)):
            loss_l = layers_loss[i] * self.weights[i]
            loss += loss_l

        return loss
        
    
    def _forward_features(self, X, label):
        X = self.normalize(X)

        self.activations = []
        self.model(X)
        X_VAL = self.activations

        layers_loss = []
        for i in range(len(X_VAL)):
            A = X_VAL[i]
            A = A.view(A.size(0), A.size(1), -1)
            d = self.disc[i](A)
            target = target_tensor = torch.full_like(d, label, dtype=torch.float32, device=A.device)
            loss = self.bceLogLoss(d, target) 
            layers_loss.append(loss)
              
        return layers_loss