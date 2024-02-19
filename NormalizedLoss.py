class NormalizedTensorLoss(nn.Module):
    def __init__(self, type="l1", reduction="mean") -> torch.Tensor:
        super().__init__()
        self.type = type
        self.reduction = reduction.lower()
        self.norm_1D = torch.nn.InstanceNorm1d(None, affine=False)
        self.norm_2D = torch.nn.InstanceNorm2d(None, affine=False)
        self.norm_3D = torch.nn.InstanceNorm3d(None, affine=False)

    def forward(self, A, B):
      if A.dim() == 1:
        A = nn.LayerNorm(A.shape[0],  elementwise_affine=False, bias=False)(A)
        B = nn.LayerNorm(B.shape[0],  elementwise_affine=False, bias=False)(B)
      elif A.dim() == 2 or A.dim() == 3:
        A = self.norm_1D(A)
        B = self.norm_1D(B)
      elif A.dim() == 4:
        A = self.norm_2D(A)
        B = self.norm_2D(B)

      elif A.dim() == 5:
        A = self.norm_3D(A)
        B = self.norm_3D(B)

      difference = A - B.detach()
      average = (A.detach().abs() + B.detach().abs() ) + 1e-9
      difference = difference / average

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