class NormalizedTensorLoss(nn.Module):
    def __init__(self, type="l1", reduction="mean") -> torch.Tensor:
        super().__init__()
        self.type = type
    
    
    def get_dims_based_on_tensor(self, tensor):
      dims = tensor.dim()
      if dims == 1:
          return (0,)
      elif dims == 2:
          return (1,)
      elif dims == 3:
        return (2,)
        #return (1, 2)
      elif dims == 4:
          return (2, 3)
      elif dims == 5:
          return (2, 3, 4)
      else:
          return None
        
        
    def min_max_normalization_combination(self, tensor_A, tensor_B, dims):
      min_val_A = torch.amin(tensor_A, dim=dims, keepdim=True)
      max_val_A = torch.amax(tensor_A, dim=dims, keepdim=True)

      min_val_B = torch.amin(tensor_B, dim=dims, keepdim=True)
      max_val_B = torch.amax(tensor_B, dim=dims, keepdim=True)

      min_val = torch.minimum(min_val_A, min_val_B)
      max_val = torch.maximum(max_val_A, max_val_B) + 1e-5

      tensor_A = (tensor_A - min_val) / (max_val - min_val)
      tensor_B = (tensor_B - min_val) / (max_val - min_val)
      return tensor_A, tensor_B


    def forward(self, A, B):
      
      A, B = self.min_max_normalization_combination(A, B, self.get_dims_based_on_tensor(A))

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