import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NormalizedTensorLoss(nn.Module):
    def __init__(self, type="l1", reduction="mean", norm="cosine") -> torch.Tensor:
        super().__init__()
        self.type = type
        self.reduction = reduction
        self.norm = norm

        if self.norm == "minmax":
         print("Using minmax")
        elif self.norm == "instancenorm":
          print("Using instancenorm")
        elif self.norm == "cosine":
          print("Using cosine")
        else:
           print("Using no normalization")
    

    def simplified_whitening(self, A, B):
      if A.dim() > 3:
            BA, C = A.size(0), A.size(1)
            A = A.view(BA, C, -1) 
            B = B.view(BA, C, -1)

      mean  = torch.mean(A, dim=-1, keepdim=True)
      var   = torch.var(A, dim=-1, keepdim=True, correction=1) + 1e-5
      std =  torch.sqrt(var)
      A = (A - mean) / std
      B = (B - mean) / std
      return A, B
    
      
    def min_max_normalization(self, tensor_A, tensor_B):
        if tensor_A.dim() > 3:
            B, C = tensor_A.size(0), tensor_A.size(1)
            tensor_A = tensor_A.view(B, C, -1)
            tensor_B = tensor_B.view(B, C, -1)

        min_val = torch.min(tensor_A, dim=-1, keepdim=True)[0]
        max_val = torch.max(tensor_A, dim=-1, keepdim=True)[0] 

        maxx = torch.abs(max_val - min_val) + 1e-2

        
        tensor_A = (tensor_A - min_val) / maxx
        tensor_B = (tensor_B - min_val) / maxx
        return tensor_A, tensor_B
    
    def cosine_similarity(self, A, B):
      if A.dim() > 3:
              A = A.view(A.size(0), A.size(1), -1)
              B = B.view(B.size(0), B.size(1), -1)

      A = F.normalize(A, p=2, dim=1)
      B = F.normalize(B, p=2, dim=1)
      
      cosine_sim = F.cosine_similarity(A, B, dim=1)
      
      loss = 1 - cosine_sim
      return loss

    def forward(self, A, B):
      if self.norm == "cosine":
        difference = self.cosine_similarity(A, B)
      
      if self.norm == "minmax":
        A, B = self.min_max_normalization(A, B)
        difference = A - B.detach()

      if self.norm == "instancenorm":
        A, B = self.simplified_whitening(A, B)
        difference = A - B.detach()

      
      

      if self.type == 'l1':
        difference = difference.abs()
      if self.type == 'l2':
        difference = difference.pow(2)

      loss = torch.tensor(0)
      if self.reduction == "mean":
        loss = difference.mean()
      if self.reduction == "sum":
        loss = difference.sum()
      if self.reduction == "none":
        loss = difference

      return loss