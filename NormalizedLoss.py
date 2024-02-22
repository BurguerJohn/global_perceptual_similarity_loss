import torch
import torch.nn as nn
import numpy as np

class NormalizedTensorLoss(nn.Module):
    def __init__(self, type="l1", reduction="mean", norm="instancenorm") -> torch.Tensor:
        super().__init__()
        self.type = type
        self.reduction = reduction
        self.norm = norm
    

    def simplified_whitening_combined(self, A, B):
      if A.dim() > 2:
            BA, C = A.size(0), A.size(1)
            A = A.view(BA, C, -1)
            B = B.view(BA, C, -1)

      mean = (torch.mean(A, dim=-1, keepdim=True) + torch.mean(B, dim=-1, keepdim=True)) * 0.5
      var = (torch.var(A, dim=-1, keepdim=True, correction=1) + torch.var(B, dim=-1, keepdim=True, correction=1)) * 0.5
      sqred = torch.sqrt(var + 1e-5)
      A = (A - mean) / sqred
      B = (B - mean) / sqred
      return A, B
      
    def min_max_normalization_combination(self, tensor_A, tensor_B):
        if tensor_A.dim() > 2:
            B, C = tensor_A.size(0), tensor_A.size(1)
            tensor_A = tensor_A.view(B, C, -1)
            tensor_B = tensor_B.view(B, C, -1)

        min_val_A = torch.min(tensor_A, dim=-1, keepdim=True)[0]
        max_val_A = torch.max(tensor_A, dim=-1, keepdim=True)[0]

        min_val_B = torch.min(tensor_B, dim=-1, keepdim=True)[0]
        max_val_B = torch.max(tensor_B, dim=-1, keepdim=True)[0]

        min_val = (min_val_A + min_val_B) * 0.5
        max_val = ((max_val_A + max_val_B) * 0.5) + 1e-5

        tensor_A = (tensor_A - min_val) / (max_val - min_val)
        tensor_B = (tensor_B - min_val) / (max_val - min_val)
        return tensor_A, tensor_B

    def forward(self, A, B):
      
      if self.norm == "minmax":
        A, B = self.min_max_normalization_combination(A, B)
      if self.norm == "instancenorm":
        A, B = self.simplified_whitening_combined(A, B)

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